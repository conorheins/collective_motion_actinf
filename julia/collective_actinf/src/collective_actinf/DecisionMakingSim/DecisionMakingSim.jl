module DecisionMakingSim

include("../GeoUtils/GeoUtils.jl")
include("../NoiseUtils/NoiseUtils.jl")
include("../AnalysisUtils/AnalysisUtils.jl")
include("../GMUtils/GMUtils.jl")
include("../SimUtils/SimUtils.jl")

using LinearAlgebra
using Statistics
using Distances

function run_decision_making_sim(r::Array{Float32,3}, v::Array{Float32,3}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any})
    """
    This function runs a single realization of multimodal schooling with external targets,
         using the new vectorized implementation across individuals.
    """

    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]
    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    dist_thr = gp_params[:dist_thr]

    # create the shift matrix
    ns_x, ndo_x = gm_params[:ns_x], gm_params[:ndo_x]
    D_shift::Array{Float32,2} = diagm(ns_x => ones(Float32,ndo_x*ns_x- ns_x));

    # create rotation matrices
    R_starts, R_ends = GeoUtils.generate_start_end_rotation_lists(gp_params[:sector_angles], reverse_flag = true)
    R_list = [R_starts, R_ends]

    noise_struct, dist_matrix, φ_hist, μ_hist, φ_t, dh_dr_self_array, empty_sector_flags, μ_t = DecisionMakingSim.initialize_history_arrays_new(gp_params, gm_params)

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        run_single_timestep(r, v, t, dist_matrix, noise_struct, N, D, gp_params,
                                gm_params, D_shift, R_list, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags)

    end # end of loop over time

    results_dict = Dict(:r => r, :v => v, :φ_hist => φ_hist, :μ_hist => μ_hist)

    return results_dict

end


function run_single_timestep(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
    gm_params::Dict{Symbol,Any}, D_shift::Matrix{Float32}, R_starts_ends, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags)
    """
    Wrapper for all the processes that happen within a single timestep of the overall active inference loop.
        This is the new vectorized version of the schooling function
    """

    # unpack global constant parameters from `gp_params`

    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    dist_thr = gp_params[:dist_thr]

    preference_idx = gp_params[:preference_idx]
    target_positions = gp_params[:target_pos]
    # unpack state space dimensionalities from `gm_params`

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    ns_φ = gm_params[:ns_φ]
    ndo_φ = gm_params[:ndo_φ]

    # unpack generative model mapping functions from `gm_params`

    g = gm_params[:sensory_func]
    ∂g∂x = gm_params[:sensory_func_prime]

    f = gm_params[:flow_func]
    ∂f∂x = gm_params[:flow_func_prime]

    # unpack precisions from `gm_params`

    𝚷_z = gm_params[:𝚷_z]
    𝚷_ω = gm_params[:𝚷_ω]

    # unpack parameters of variational belief updating from `gm_params`

    num_iter = gm_params[:num_iter]
    κ_μ = gm_params[:κ_μ]

    # unpack action-updating-related parameters from `gm_params`
    κ_a = gm_params[:κ_a]

    noise_samples_action = reshape(NoiseUtils.get_samples(noise_struct, D*N), (D, N))

    ####### UPDATE GENERATIVE PROCESS #######

    r_t, v_past = SimUtils.update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    sector_idx = GeoUtils.assign_visual_field_ids_new(R_starts_ends[1], R_starts_ends[2], r_t, v_past, dist_thr, dist_matrix)

    all_sector_h, all_sector_h_prime, all_dh_dr_self = GeoUtils.calculate_sector_hidden_states_vectorized(dist_matrix, sector_idx, r_t, v_past)

    _, _, ∂tdist∂r_per_target = GeoUtils.calculate_dists_to_targets(r_t, v_past, preference_idx, target_positions)

    ####### SAMPLE OBSERVATIONS FROM GENERATIVE PROCESS #######
    noise_samples_φ = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))

    φ_t = SimUtils.get_observations_vectorized(all_sector_h, all_sector_h_prime, noise_samples_φ, gp_params) # N.B. Need to change φ_t that is passed into the run_single_timestep_new function at the top so that it's alsways in vectorized form (not in ns_φ X ndo_φ X N )

    # nullify the empty sectors
    empty_sector_mask = vcat([ hcat(all_sector_h...), hcat(all_sector_h_prime...)]...) .== 0f0
    φ_t_h[empty_sector_mask] .= 0f0

    ####### UPDATE BELIEFS USING PREDICTIVE CODING #######
    if t == 2
        μ_t[1:(ns_φ*ndo_φ),:] =  φ_t # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
    else
        μ_t = copy(μ_hist[:,:,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
    end

    μ_t, ε_z = SimUtils.run_belief_updating_vectorized(μ_t, φ_t, gm_params, gp_params, D_shift)

    ####### UPDATE ACTIONS USING ACTIVE INFERENCE  #######
    ∂F∂v = SimUtils.update_action_new(ε_z, gm_params, all_sector_h_prime, all_dh_dr_self)

    v[:,:,t] -= (κ_a .* ∂F∂v)

    # add in the influence of the target(s)
    for target_idx in 1:length(preference_idx)
        v[:,preference_idx[target_idx], t] -= (κ_a .* ∂tdist∂r_per_target[target_idx])
    end

    v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities
    # x_hist[:,:,:,t] .= copy(x_t)
    φ_hist[:,:,t] .= copy(φ_t)
    μ_hist[:,:,t] .= copy(μ_t)

    return

end

function generate_targets(origin::Array{Float32,1}, distance::Float32, theta::Float32)::Vector{Array{Float32,2}}
    """
    This function return the position vectors of two points that are `distance` units away from an initial point `origin`
        and subtending angle `theta` (in degrees) from `origin`.
    Arguments
    ============
        `origin` [Array{Float32,1}]: size (D,) position vector of the origin -- this could be e.g. the centroid position of the school
        `distance` [Float32]: scalar that determines the distance between origin and each of the two target points
        `theta` [Float32]: scalar that determines the angle (in degrees) between the two targets and `origin`
    Returns
    ===========
        `target_pos` [Vector{Array{Float32,2}}]- the positions of both target positions, reshaped into respectively-size (D,1) column vectors vectors and collected in an outer Vector{} array
    """
    D = length(origin);

    h = distance * cosd(0.5f0*theta);
    x = distance * sind(0.5f0*theta);

    # displacement_vector = rand(Float32,D);
    displacement_vector = [0f0, 1f0]
    displacement_vector ./= sqrt(sum(displacement_vector.^2))
    midpoint_base = origin .+ (h .* displacement_vector);

    perpendicular_line = midpoint_base .- origin;

    line_of_base = [-perpendicular_line[2], perpendicular_line[1]] ./ norm(perpendicular_line)

    target1 = midpoint_base .+ (x .* line_of_base);
    target2 = midpoint_base .- (x .* line_of_base);

    target_pos = reshape.( [target1, target2], D, 1)

    return target_pos

end

function generate_default_gp_params(N, T, D; ns_φ = 4, ndo_φ = 2, dt = 0.01f0, dist_thr = 7f0, sector_angles = [120f0, 60f0, 0f0, 360f0 - 60f0, 360f0 - 120f0],
                                    z_gp_h = 0.1f0, z_action = 0.05f0, z_gp_tdist = 0.1f0, α_g = 10f0, b = 3.5f0, target_dist = 5.0f0, target_angle = 60f0, preference_idx = nothing)

    t_axis = 0:dt:T; # time axis in seconds
    T_sim = length(t_axis); # length of time-axis in number of samples

    EM_scalar = sqrt(dt)
    z_gp_h = EM_scalar .* z_gp_h * ones(Float32,ns_φ,ndo_φ)
    z_action = EM_scalar .* (z_action.* ones(Float32,D))
    z_gp_tdist = EM_scalar * z_gp_tdist

    # generate the sensory transformation functions of the generative process, given the α_g and b parameter defined in the inputs
    g(x) = x ./ (1f0 .+ exp.(α_g .* (x .- b)));

    function ∂g∂x(x)
        output = 1f0 ./ (1f0 .+ exp.(α_g .* (x .- b))) .- (α_g .* x .* exp.(α_g .* (x .- b))) ./ (exp.(α_g .* (x .- b)) .+ 1f0).^2;

        if typeof(output) == Float32
            if abs(output) < 1.0f-8 || isnan(output)
                output = 0f0
            end
        else
            output[ (abs.(output) .< 1.0f-8) .| isnan.(output) ] .= 0f0; # prevent numerical underflow
        end

        return output
    end


    # sampling functions of the generative process
    sampling_func_φ(h, noise) = g.(h) .+ z_gp_h[:,1] .* noise[:,1]
    h_by_hprime(h,hprime) = ∂g∂x.(h) .* hprime
    sampling_func_φprime(h_x_hprime,noise) = h_x_hprime .+ z_gp_h[:,2] .* noise[:,2]

    sampling_func_tdist(state, noise) = state .+ (z_gp_tdist .* noise[1,:])
    sampling_func_tdistprime(state, noise) = state .+ (z_gp_tdist .* noise[2,:])

    if isnothing(preference_idx)
        # by default, have 30% of the school be informed, and then split them up (half/half) into target1 vs target2 preferring individuals

        num_informed = round(Int64, 0.3 * N)
        informed_idx = rand(1:N, N)[1:num_informed]

        num_target1 = round(Int64, num_informed / 2)

        preference_idx = [informed_idx[1:num_target1], informed_idx[num_target1+1:end]]
    end

    target_pos = generate_targets(zeros(Float32, D), target_dist, target_angle)

    if length(preference_idx) == 1
        target_pos = [target_pos[1]]
    end

    gp_params = Dict(:N => N, :D => D, :T_sim => T_sim, :dt => dt, :dist_thr => dist_thr,
                    :sector_angles => sector_angles, :EM_scalar => EM_scalar, :z_gp_h => z_gp_h, :z_action => z_action, :z_gp_tdist => z_gp_tdist,
                    :sampling_func_φ => sampling_func_φ, :h_by_hprime => h_by_hprime, :sampling_func_φprime => sampling_func_φprime, :sampling_func_tdist => sampling_func_tdist,
                    :sampling_func_tdistprime => sampling_func_tdistprime, :target_pos => target_pos, :preference_idx => preference_idx)

    return gp_params
end

function generate_default_gp_params_linear_g(N, T, D; ns_φ = 4, ndo_φ = 2, dt = 0.01f0, dist_thr = 7f0, sector_angles = [120f0, 60f0, 0f0, 360f0 - 60f0, 360f0 - 120f0],
                                    z_gp_h = 0.1f0, z_action = 0.05f0, z_gp_tdist = 0.1f0, target_dist = 5.0f0, target_angle = 60f0, preference_idx = nothing)

    t_axis = 0:dt:T; # time axis in seconds
    T_sim = length(t_axis); # length of time-axis in number of samples

    EM_scalar = sqrt(dt)
    z_gp_h = EM_scalar .* z_gp_h * ones(Float32,ns_φ,ndo_φ)
    z_action = EM_scalar .* (z_action.* ones(Float32,D))
    z_gp_tdist = EM_scalar * z_gp_tdist

    function g(x)
        return x
    end

    function ∂g∂x(x)
        return 1f0
    end

    # sampling functions of the generative process
    sampling_func_φ(h, noise) = g.(h) .+ z_gp_h[:,1] .* noise[:,1]
    h_by_hprime(h,hprime) = ∂g∂x.(h) .* hprime
    sampling_func_φprime(h_x_hprime,noise) = h_x_hprime .+ z_gp_h[:,2] .* noise[:,2]

    sampling_func_tdist(state, noise) = state .+ (z_gp_tdist .* noise[1,:])
    sampling_func_tdistprime(state, noise) = state .+ (z_gp_tdist .* noise[2,:])

    if isnothing(preference_idx)
        # by default, have 30% of the school be informed, and then split them up (half/half) into target1 vs target2 preferring individuals

        num_informed = round(Int64, 0.3 * N)
        informed_idx = rand(1:N, N)[1:num_informed]

        num_target1 = round(Int64, num_informed / 2)

        preference_idx = [informed_idx[1:num_target1], informed_idx[num_target1+1:end]]
    end

    target_pos = generate_targets(zeros(Float32, D), target_dist, target_angle)

    if length(preference_idx) == 1
        target_pos = [target_pos[1]]
    end

    gp_params = Dict(:N => N, :D => D, :T_sim => T_sim, :dt => dt, :dist_thr => dist_thr,
                    :sector_angles => sector_angles, :EM_scalar => EM_scalar, :z_gp_h => z_gp_h, :z_action => z_action, :z_gp_tdist => z_gp_tdist,
                    :sampling_func_φ => sampling_func_φ, :h_by_hprime => h_by_hprime, :sampling_func_φprime => sampling_func_φprime, :sampling_func_tdist => sampling_func_tdist,
                    :sampling_func_tdistprime => sampling_func_tdistprime, :target_pos => target_pos, :preference_idx => preference_idx)

    return gp_params
end



function initialize_history_arrays_new(gp_params::Dict{Symbol, Any}, gm_params::Dict{Symbol, Any})
    """
    New version of initializing the arrays for storing realization-specific data
    """
    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]

    # unpack state space dimensionalities from `gm_params`
    ns_x = gm_params[:ns_x] # number of hidden states
    ndo_x = gm_params[:ndo_x] # generalised orders of hidden states
    ns_φ = gm_params[:ns_φ]  # number of observation dimensions
    ndo_φ = gm_params[:ndo_φ] # generalised orders of observation dimensions

    # create noise structure
    noise_struct = NoiseUtils.NoiseStruct(num_samples = (T_sim*N*D*ns_φ*ndo_φ*N) )

    # create distance matrix for storing instantaneous distances between agents
    dist_matrix::Array{Float32,2} = zeros(Float32,N,N)

    φ_hist = zeros(Float32, (ns_φ*ndo_φ), N, T_sim) # history of observations

    μ_hist    = zeros(Float32,(ns_x*ndo_x), N, T_sim) # history of beliefs

    φ_t = zeros(Float32, (ns_φ*ndo_φ), N); # running cache for observations

    dh_dr_self_array = zeros(Float32, D, ns_x, N); # running cache for partial derivatives of hidden states with respect to position of focal particle
    empty_sector_flags = falses(ns_x, ndo_x, N); # running cache for flags that indicate absence of observations for each modality & dyamical order & individual

    μ_t = zeros(Float32,ns_x*ndo_x, N) # running cache for storing instantaneous beliefs for a given individual

    return noise_struct, dist_matrix, φ_hist, μ_hist, φ_t, dh_dr_self_array, empty_sector_flags, μ_t

end


function run_decision_making_sim_v2(r::Array{Float32,3}, v::Array{Float32,3}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any})
    """
    This function runs a single realization of multimodal schooling with external targets,
         using the new vectorized implementation across individuals.
    """

    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]
    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    dist_thr = gp_params[:dist_thr]

    # create the shift matrix
    ns_x, ndo_x = gm_params[:ns_x], gm_params[:ndo_x]
    D_shift_tdist::Array{Float32,2} = diagm(1 => ones(Float32,ndo_x - 1));

    # create rotation matrices
    R_starts, R_ends = GeoUtils.generate_start_end_rotation_lists(gp_params[:sector_angles], reverse_flag = true)
    R_list = [R_starts, R_ends]

    noise_struct, dist_matrix, φ_hist, μ_hist, dh_dr_self_array, empty_sector_flags, μ_t = DecisionMakingSim.initialize_history_arrays_v2(gp_params, gm_params, r, v)

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        run_single_timestep_v2(r, v, t, dist_matrix, noise_struct, N, D, gp_params,
                                gm_params, D_shift_tdist, R_list, dh_dr_self_array, φ_hist, μ_hist, μ_t, empty_sector_flags)

    end # end of loop over time

    results_dict = Dict(:r => r, :v => v, :φ_hist => φ_hist, :μ_hist => μ_hist)

    return results_dict

end

function initialize_history_arrays_v2(gp_params::Dict{Symbol, Any}, gm_params::Dict{Symbol, Any}, r::Array{Float32, 3}, v::Array{Float32, 3})
    """
    New version of initializing the arrays for storing realization-specific data
    """
    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]

    # unpack state space dimensionalities from `gm_params`
    ns_x = gm_params[:ns_x] # number of hidden states
    ndo_x = gm_params[:ndo_x] # generalised orders of hidden states
    ns_φ = gm_params[:ns_φ]  # number of observation dimensions
    ndo_φ = gm_params[:ndo_φ] # generalised orders of observation dimensions

    # create noise structure
    noise_struct = NoiseUtils.NoiseStruct(num_samples = Int64(10e4) )

    # create distance matrix for storing instantaneous distances between agents
    dist_matrix::Array{Float32,2} = zeros(Float32,N,N)

    φ_hist_targets = [zeros(Float32, ndo_φ, length(idx), T_sim) for idx in gp_params[:preference_idx]]
    φ_hist = [zeros(Float32, (ns_φ*ndo_φ), N, T_sim), φ_hist_targets] # history of observations of distances in each sector and history of observations of distance to target

    μ_hist_targets = [zeros(Float32, ndo_x, length(idx), T_sim) for idx in gp_params[:preference_idx]]
    μ_hist    = [zeros(Float32,(ns_x*ndo_x), N, T_sim), μ_hist_targets] # history of beliefs about distance in each sector, and history of beliefs about distance to target

    tdist_per_target, tdist_prime_per_target, _ = GeoUtils.calculate_dists_to_targets(r[:,:,1], v[:,:,1], gp_params[:preference_idx], gp_params[:target_pos])

    for target_i in 1:length(gp_params[:target_pos])
        φ_hist_targets[target_i][1,:,1] = copy(tdist_per_target[target_i])
        φ_hist_targets[target_i][2,:,1] = copy(tdist_prime_per_target[target_i])

        μ_hist_targets[target_i][1,:,1] = copy(tdist_per_target[target_i])
        μ_hist_targets[target_i][2,:,1] = copy(tdist_prime_per_target[target_i])
    end

    dh_dr_self_array = zeros(Float32, D, ns_x, N); # running cache for partial derivatives of hidden states with respect to position of focal particle
    empty_sector_flags = falses(ns_x, ndo_x, N); # running cache for flags that indicate absence of observations for each modality & dyamical order & individual

    μ_t_targets = [copy(μ_hist_targets[target_i][:,:,1]) for target_i in 1:length(gp_params[:target_pos])]
    μ_t = [zeros(Float32,ns_x*ndo_x, N),  μ_t_targets] # running cache for storing instantaneous beliefs about distance in each sector and about the distnace to target, for a given timestep

    return noise_struct, dist_matrix, φ_hist, μ_hist, dh_dr_self_array, empty_sector_flags, μ_t
end


function run_single_timestep_v2(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
    gm_params::Dict{Symbol,Any}, D_shift_tdist::Matrix{Float32},R_starts_ends, dh_dr_self_array, φ_hist, μ_hist, μ_t, empty_sector_flags)
    """
    Wrapper for all the processes that happen within a single timestep of the overall active inference loop.
        This is the new vectorized version of the schooling function
    """

    # unpack global constant parameters from `gp_params`

    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    dist_thr = gp_params[:dist_thr]

    preference_idx = gp_params[:preference_idx]
    target_positions = gp_params[:target_pos]
    # unpack state space dimensionalities from `gm_params`

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    ns_φ = gm_params[:ns_φ]
    ndo_φ = gm_params[:ndo_φ]

    # unpack generative model mapping functions from `gm_params`

    # g = gm_params[:sensory_func]
    # ∂g∂x = gm_params[:sensory_func_prime]
    #
    # f = gm_params[:flow_func]
    # ∂f∂x = gm_params[:flow_func_prime]

    # unpack precisions from `gm_params`

    𝚷_z = gm_params[:𝚷_z]
    𝚷_ω = gm_params[:𝚷_ω]

    # unpack parameters of variational belief updating from `gm_params`

    num_iter = gm_params[:num_iter]
    κ_μ = gm_params[:κ_μ]

    # unpack action-updating-related parameters from `gm_params`
    κ_a = gm_params[:κ_a]

    noise_samples_action = reshape(NoiseUtils.get_samples(noise_struct, D*N), (D, N))

    ####### UPDATE GENERATIVE PROCESS #######

    r_t, v_past = SimUtils.update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    sector_idx = GeoUtils.assign_visual_field_ids_new(R_starts_ends[1], R_starts_ends[2], r_t, v_past, dist_thr, dist_matrix)

    all_sector_h, all_sector_h_prime, all_dh_dr_self = GeoUtils.calculate_sector_hidden_states_vectorized(dist_matrix, sector_idx, r_t, v_past)

    tdist_per_target, tdist_prime_per_target, ∂tdist∂r_per_target = GeoUtils.calculate_dists_to_targets(r_t, v_past, preference_idx, target_positions)

    ####### SAMPLE OBSERVATIONS FROM GENERATIVE PROCESS #######

    # first get the sector-wise distance observations
    noise_samples_φ_h = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))
    φ_t_h = SimUtils.get_observations_vectorized(all_sector_h, all_sector_h_prime, noise_samples_φ_h, gp_params) # N.B. Need to change φ_t that is passed into the run_single_timestep_new function at the top so that it's alsways in vectorized form (not in ns_φ X ndo_φ X N )

    # nullify the empty sectors
    empty_sector_mask = vcat([ hcat(all_sector_h...), hcat(all_sector_h_prime...)]...) .== 0f0
    φ_t_h[empty_sector_mask] .= 0f0

    # now get the target obseravtions

    # get noise samples for the tdist (state and velocity) for each target
    noise_samples_φ_target = [reshape(NoiseUtils.get_samples(noise_struct, ndo_φ*length(idx)), (ndo_φ, length(idx))) for idx in preference_idx]

    # create the observations by integrating the noise samples with the observation function of the generative process
    φ_tdist = map(gp_params[:sampling_func_tdist], tdist_per_target, noise_samples_φ_target)
    φ_tdistprime = map(gp_params[:sampling_func_tdistprime], tdist_prime_per_target, noise_samples_φ_target)

    # aggregate the noise samples into a vector of matrices, one per target
    φ_t_target = map((state, velocity) -> hcat(state, velocity)', φ_tdist, φ_tdistprime)

    ####### UPDATE BELIEFS USING PREDICTIVE CODING #######
    if t == 2
        μ_t[1][1:(ns_φ*ndo_φ),:] =  φ_t_h # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
        for target_id in 1:length(preference_idx)
            μ_t[2][target_id][1:ndo_φ,:] = φ_t_target[target_id]
        end
    else
        μ_t[1] = copy(μ_hist[1][:,:,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
        for target_id in 1:length(preference_idx)
            μ_t[2][target_id] = copy(μ_hist[2][target_id][:,:,t-1]);
        end
    end


    μ_t[1], ε_z_h = SimUtils.run_belief_updating_vectorized(μ_t[1], φ_t_h, gm_params, gp_params)

    μ_t[2], ε_z_tdist =  SimUtils.run_belief_updating_tdist(μ_t[2], φ_t_target, gm_params, gp_params, D_shift_tdist)

    ####### UPDATE ACTIONS USING ACTIVE INFERENCE  #######
    ∂F∂v = SimUtils.update_action_new(ε_z_h, gm_params, all_sector_h_prime, all_dh_dr_self)

    v[:,:,t] -= (κ_a .* ∂F∂v)

    # add in the influence of the target(s)
    for target_id in 1:length(preference_idx)
        ∂F∂_tdistprime = ε_z_tdist[target_id][2,:]
        ∂tdistprime_∂v = ∂tdist∂r_per_target[target_id]
        ∂F∂v = ∂F∂_tdistprime' .* ∂tdistprime_∂v
        v[:,preference_idx[target_id], t] -= (κ_a .* ∂F∂v)
    end

    v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities

    φ_hist[1][:,:,t] .= copy(φ_t_h)
    μ_hist[1][:,:,t] .= copy(μ_t[1])
    for target_id in 1:length(preference_idx)
        φ_hist[2][target_id][:,:,t] = copy(φ_t_target[target_id])
        μ_hist[2][target_id][:,:,t] = copy(μ_t[2][target_id])
    end

    return

end

function run_decision_making_sim_v3(r::Array{Float32,3}, v::Array{Float32,3}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any})
    """
    This function runs a single realization of multimodal schooling with external targets,
         using the new vectorized implementation across individuals with precision learning using the Beta updates from Baouimy et al. 2022
    """

    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]
    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    dist_thr = gp_params[:dist_thr]

    # create the shift matrix
    ns_x, ndo_x = gm_params[:ns_x], gm_params[:ndo_x]
    D_shift::Array{Float32,2} = diagm(ns_x => ones(Float32,ndo_x*ns_x- ns_x));
    D_shift_tdist::Array{Float32,2} = diagm(1 => ones(Float32,ndo_x - 1));

    # create rotation matrices
    R_starts, R_ends = GeoUtils.generate_start_end_rotation_lists(gp_params[:sector_angles], reverse_flag = true)
    R_list = [R_starts, R_ends]

    noise_struct, dist_matrix, φ_hist, μ_hist, Γ_z_target_hist, dh_dr_self_array, μ_t, Γ_z_t = DecisionMakingSim.initialize_history_arrays_v3(gp_params, gm_params, r, v)

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        run_single_timestep_v3(r, v, t, dist_matrix, noise_struct, N, D, gp_params,
                                gm_params, D_shift, D_shift_tdist, R_list, dh_dr_self_array, φ_hist, μ_hist, Γ_z_target_hist, μ_t, Γ_z_t)

    end # end of loop over time

    results_dict = Dict(:r => r, :v => v, :φ_hist => φ_hist, :μ_hist => μ_hist, :Γ_z_target_hist => Γ_z_target_hist)

    return results_dict

end


function initialize_history_arrays_v3(gp_params::Dict{Symbol, Any}, gm_params::Dict{Symbol, Any}, r::Array{Float32, 3}, v::Array{Float32, 3})
    """
    New version of initializing the arrays for storing realization-specific data. This version also has new arrays for storing history of precisions and
        pre-initializes the mus to be based on true values
    """
    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]

    # unpack state space dimensionalities from `gm_params`
    ns_x = gm_params[:ns_x] # number of hidden states
    ndo_x = gm_params[:ndo_x] # generalised orders of hidden states
    ns_φ = gm_params[:ns_φ]  # number of observation dimensions
    ndo_φ = gm_params[:ndo_φ] # generalised orders of observation dimensions

    # create noise structure
    noise_struct = NoiseUtils.NoiseStruct(num_samples = Int64(10e4) )

    # create distance matrix for storing instantaneous distances between agents
    dist_matrix::Array{Float32,2} = zeros(Float32,N,N)

    φ_hist_targets = [zeros(Float32, ndo_φ, length(idx), T_sim) for idx in gp_params[:preference_idx]]
    φ_hist = [zeros(Float32, (ns_φ*ndo_φ), N, T_sim), φ_hist_targets] # history of observations of distances in each sector and history of observations of distance to target

    μ_hist_targets = [zeros(Float32, ndo_x, length(idx), T_sim) for idx in gp_params[:preference_idx]]
    μ_hist    = [zeros(Float32,(ns_x*ndo_x), N, T_sim), μ_hist_targets] # history of beliefs about distance in each sector, and history of beliefs about distance to target

    tdist_per_target, tdist_prime_per_target, _ = GeoUtils.calculate_dists_to_targets(r[:,:,1], v[:,:,1], gp_params[:preference_idx], gp_params[:target_pos])

    for target_i in 1:length(gp_params[:target_pos])
        φ_hist_targets[target_i][1,:,1] = copy(tdist_per_target[target_i])
        φ_hist_targets[target_i][2,:,1] = copy(tdist_prime_per_target[target_i])

        μ_hist_targets[target_i][1,:,1] = copy(tdist_per_target[target_i])
        μ_hist_targets[target_i][2,:,1] = copy(tdist_prime_per_target[target_i])
    end

    α_init_scalars = gm_params[:β_scalar] .* diag(gm_params[:𝚷_z_tdist])
    # history of alpha and beta parameters of Gamma posterior over precision. One matrix of alphas and betas per target
    Γ_z_target_hist = [ [α_init_scalars .* ones(Float32, ndo_φ, length(idx), T_sim), gm_params[:β_scalar] .* ones(Float32, ndo_φ, length(idx), T_sim)] for idx in gp_params[:preference_idx]]

    dh_dr_self_array = zeros(Float32, D, ns_x, N); # running cache for partial derivatives of hidden states with respect to position of focal particle

    μ_t_targets = [copy(μ_hist_targets[target_i][:,:,1]) for target_i in 1:length(gp_params[:target_pos])]
    μ_t = [zeros(Float32,ns_x*ndo_x, N),  μ_t_targets] # running cache for storing instantaneous beliefs about distance in each sector and about the distnace to target, for a given timestep

    # current setting of alpha and beta parameters of Gamma posterior over precision. One matrix of alphas and betas per target
    Γ_z_t = [ [α_init_scalars .* ones(Float32, ndo_φ, length(idx)), gm_params[:β_scalar] .* ones(Float32, ndo_φ, length(idx))] for idx in gp_params[:preference_idx]]

    return noise_struct, dist_matrix, φ_hist, μ_hist, Γ_z_target_hist, dh_dr_self_array, μ_t, Γ_z_t
end



function run_single_timestep_v3(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
    gm_params::Dict{Symbol,Any}, D_shift::Matrix{Float32}, D_shift_tdist::Matrix{Float32},R_starts_ends, dh_dr_self_array, φ_hist, μ_hist, Γ_z_target_hist, μ_t, Γ_z_t)
    """
    Wrapper for all the processes that happen within a single timestep of the overall active inference loop.
        This is the new vectorized version of the schooling function
    """

    # unpack global constant parameters from `gp_params`

    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    dist_thr = gp_params[:dist_thr]

    preference_idx = gp_params[:preference_idx]
    target_positions = gp_params[:target_pos]
    # unpack state space dimensionalities from `gm_params`

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    ns_φ = gm_params[:ns_φ]
    ndo_φ = gm_params[:ndo_φ]

    # unpack action-updating-related parameters from `gm_params`
    κ_a = gm_params[:κ_a]

    noise_samples_action = reshape(NoiseUtils.get_samples(noise_struct, D*N), (D, N))

    ####### UPDATE GENERATIVE PROCESS #######

    r_t, v_past = SimUtils.update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    sector_idx = GeoUtils.assign_visual_field_ids_new(R_starts_ends[1], R_starts_ends[2], r_t, v_past, dist_thr, dist_matrix)

    all_sector_h, all_sector_h_prime, all_dh_dr_self = GeoUtils.calculate_sector_hidden_states_vectorized(dist_matrix, sector_idx, r_t, v_past)

    tdist_per_target, tdist_prime_per_target, ∂tdist∂r_per_target = GeoUtils.calculate_dists_to_targets(r_t, v_past, preference_idx, target_positions)

    ####### SAMPLE OBSERVATIONS FROM GENERATIVE PROCESS #######

    # first get the sector-wise distance observations
    noise_samples_φ_h = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))
    φ_t_h = SimUtils.get_observations_vectorized(all_sector_h, all_sector_h_prime, noise_samples_φ_h, gp_params) # N.B. Need to change φ_t that is passed into the run_single_timestep_new function at the top so that it's alsways in vectorized form (not in ns_φ X ndo_φ X N )

    # nullify the empty sectors
    empty_sector_mask = vcat([ hcat(all_sector_h...), hcat(all_sector_h_prime...)]...) .== 0f0
    φ_t_h[empty_sector_mask] .= 0f0

    # now get the target obseravtions

    # get noise samples for the tdist (state and velocity) for each target
    noise_samples_φ_target = [reshape(NoiseUtils.get_samples(noise_struct, ndo_φ*length(idx)), (ndo_φ, length(idx))) for idx in preference_idx]

    # create the observations by integrating the noise samples with the observation function of the generative process
    φ_tdist = map(gp_params[:sampling_func_tdist], tdist_per_target, noise_samples_φ_target)
    φ_tdistprime = map(gp_params[:sampling_func_tdistprime], tdist_prime_per_target, noise_samples_φ_target)

    # aggregate the noise samples into a vector of matrices, one per target
    φ_t_target = map((state, velocity) -> hcat(state, velocity)', φ_tdist, φ_tdistprime)

    ####### UPDATE BELIEFS USING PREDICTIVE CODING #######
    if t == 2
        μ_t[1][1:(ns_φ*ndo_φ),:] =  φ_t_h # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
        for target_id in 1:length(preference_idx)
            μ_t[2][target_id][1:ndo_φ,:] = φ_t_target[target_id]
        end
    else
        μ_t[1] = copy(μ_hist[1][:,:,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
        for target_id in 1:length(preference_idx)
            μ_t[2][target_id] = copy(μ_hist[2][target_id][:,:,t-1]);
        end
    end

    μ_t[1], ε_z_h = SimUtils.run_belief_updating_vectorized(μ_t[1], φ_t_h, gm_params, gp_params, D_shift)

    μ_t[2], ε_z_tdist, Γ_z_t = SimUtils.run_belief_updating_tdist_withGammaPlearning(μ_t[2], φ_t_target, Γ_z_t, gm_params, gp_params, D_shift_tdist)

    ####### UPDATE ACTIONS USING ACTIVE INFERENCE  #######
    ∂F∂v = SimUtils.update_action_new(ε_z_h, gm_params, all_sector_h_prime, all_dh_dr_self)

    v[:,:,t] -= (κ_a .* ∂F∂v)

    # add in the influence of the target(s)
    for target_id in 1:length(preference_idx)
        ∂F∂_tdistprime = ε_z_tdist[target_id][2,:]
        ∂tdistprime_∂v = ∂tdist∂r_per_target[target_id]
        ∂F∂v = ∂F∂_tdistprime' .* ∂tdistprime_∂v
        v[:,preference_idx[target_id], t] -= (κ_a .* ∂F∂v)
    end

    v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities

    φ_hist[1][:,:,t] .= copy(φ_t_h)
    μ_hist[1][:,:,t] .= copy(μ_t[1])
    for target_id in 1:length(preference_idx)
        φ_hist[2][target_id][:,:,t] = copy(φ_t_target[target_id])
        μ_hist[2][target_id][:,:,t] = copy(μ_t[2][target_id])
        Γ_z_target_hist[target_id][1][:,:,t] = copy(Γ_z_t[target_id][1])
        Γ_z_target_hist[target_id][2][:,:,t] = copy(Γ_z_t[target_id][2])
    end

    return

end



end
