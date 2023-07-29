module SimUtils

include("../GeoUtils/GeoUtils.jl")
include("../NoiseUtils/NoiseUtils.jl")
include("../AnalysisUtils/AnalysisUtils.jl")
include("../GMUtils/GMUtils.jl")

using LinearAlgebra
using Statistics
using Distances
using BlockDiagonals
# using NNLib

function initialise_positions_velocities(T_sim::Int64, D::Int64, N::Int64;
    position_bounds::Array{Float32,1}=[-1f0, 1f0], velocity_bounds::Array{Float32,1}=[-1f0, 1f0])::Tuple{Array{Float32,3},Array{Float32,3}}
    """
    This function initialises an array of positions and velocities `r` and `v` for
    a multi-particle simulation, using parameters of the simulation and upper/lower bounds on the
    values that the positions and velocities can take.
    INPUTS:
        `T_sim`::Int64 - total length of time of the simulation, in integer number of samples
        `D`::Int64 - number of spatial dimensions (only 2 supported for now)
        `N`::Int64 - number of particles in simulation
        `position_bounds`::Array{Float32,1} - optional parameter that gives the upper and lower bounds of the
        initial positions.
        `velocity_bounds`::Array{Float32,1} - optional parameter that gives the upper and lower bounds of the
        initial velocities.
    OUTPUTS:
        `r`::Array{Float32,3} - a size (D, N, T_sim) array that stores the spatial positions of all the particles over
        simulation time. r[:,:,1] will be filled out with initialised positions of all the particles.
        `v`::Array{Float32,3} - a size (D, N, T_sim) array that stores the D-dimensional velocities of all the particles over
        simulation time. v[:,:,1] will be filled out with initialised velocities of all the particles. The velocities will also
        be normalized to unit length.
    """

    r::Array{Float32,3}= zeros(Float32, D, N, T_sim);

    r[:,:,1] = position_bounds[1] .+ (position_bounds[2]- position_bounds[1]) .*rand(Float32,D,N);

    v::Array{Float32,3}= zeros(Float32, D, N, T_sim);
    v[:,:,1] = velocity_bounds[1] .+ (velocity_bounds[2]- velocity_bounds[1]) .*rand(Float32,D,N);
    v[:,:,1] = v[:,:,1] ./ sqrt.(sum(v[:,:,1].^2,dims=1)); # normalize velocities to unit length

    return r, v
end

function generate_rotation_matrices(sector_angles::Array{Float32,1})::Array{Array{Real,2},1}
    """
    This function generates a set of rotation matrices corresponding to the rotations required to generate
    the sector-boundary vectors corresponding to the edges of visual zones. This is accomplished multiplying each matrix
    with the heading-direction vector of a particle, whose heading vector is assumed to have angle 0s (parallel to the axis of rotation)
    """

    R_matrices = Array{Array{Real,2},1}(undef, length(sector_angles));

    for s_i = 1:length(sector_angles)
        R_matrices[s_i] = [cosd(sector_angles[s_i]) -sind(sector_angles[s_i]); sind(sector_angles[s_i]) cosd(sector_angles[s_i]) ]
    end

    return R_matrices

end

function run_simulation_new(r::Array{Float32,3}, v::Array{Float32,3}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any})
    """
    This function runs a single realization of multimodal schooling, using the new vectorized implementation across individuals.
    """

    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]
    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

    # create the shift matrix
    ns_x, ndo_x = gm_params[:ns_x], gm_params[:ndo_x]
    # D_shift::Array{Float32,2} = diagm(ns_x => ones(Float32,ndo_x*ns_x- ns_x));

    # create rotation matrices
    R_starts, R_ends = GeoUtils.generate_start_end_rotation_lists(gp_params[:sector_angles], reverse_flag = true)
    R_list = [R_starts, R_ends]

    noise_struct, dist_matrix, φ_hist, μ_hist, φ_t, dh_dr_self_array, empty_sector_flags, μ_t = initialize_history_arrays_new(gp_params, gm_params)

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        # run_single_timestep_new(r, v, t, dist_matrix, noise_struct, N, D, gp_params,
        #                         gm_params, D_shift, R_list, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags)
        run_single_timestep_new(r, v, t, dist_matrix, noise_struct, N, D, gp_params,
                                gm_params, R_list, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags)

    end # end of loop over time

    results_dict = Dict(:r => r, :v => v, :φ_hist => φ_hist, :μ_hist => μ_hist)

    return results_dict

end # end of run_simulation function

# function run_single_timestep_new(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
#     gm_params::Dict{Symbol,Any}, D_shift::Matrix{Float32}, R_starts_ends, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags)
function run_single_timestep_new(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
        gm_params::Dict{Symbol,Any}, R_starts_ends, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags)
    """
    Wrapper for all the processes that happen within a single timestep of the overall active inference loop.
        This is the new vectorized version of the schooling function
    """

    # unpack global constant parameters from `gp_params`

    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

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

    r_t, v_past = update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    sector_idx = GeoUtils.assign_visual_field_ids_new(R_starts_ends[1], R_starts_ends[2], r_t, v_past, dist_thr, dist_matrix)

    all_sector_h, all_sector_h_prime, all_dh_dr_self = GeoUtils.calculate_sector_hidden_states_vectorized(dist_matrix, sector_idx, r_t, v_past)

    # x_t = permutedims(cat(hcat(all_sector_h...), hcat(all_sector_h_prime...), dims = 3), (1, 3, 2))

    ####### SAMPLE OBSERVATIONS FROM GENERATIVE PROCESS #######
    noise_samples_φ = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))

    φ_t = get_observations_vectorized(all_sector_h, all_sector_h_prime, noise_samples_φ, gp_params) # N.B. Need to change φ_t that is passed into the run_single_timestep_new function at the top so that it's alsways in vectorized form (not in ns_φ X ndo_φ X N )

    # nullify the empty sectors
    empty_sector_mask = vcat([ hcat(all_sector_h...), hcat(all_sector_h_prime...)]...) .== 0f0
    φ_t[empty_sector_mask] .= 0f0

    ####### UPDATE BELIEFS USING PREDICTIVE CODING #######
    if t == 2
        μ_t[1:(ns_φ*ndo_φ),:] =  φ_t # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
    else
        μ_t = copy(μ_hist[:,:,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
    end

    # μ_t, ε_z = run_belief_updating_vectorized(μ_t, φ_t, gm_params, gp_params, D_shift)
    μ_t, ε_z = run_belief_updating_vectorized(μ_t, φ_t, gm_params, gp_params)

    ####### UPDATE ACTIONS USING ACTIVE INFERENCE  #######
    ∂F∂v = update_action_new(ε_z, gm_params, all_sector_h_prime, all_dh_dr_self)

    v[:,:,t] -= (κ_a .* ∂F∂v)

    v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities
    # x_hist[:,:,:,t] .= copy(x_t)
    φ_hist[:,:,t] .= copy(φ_t)
    μ_hist[:,:,t] .= copy(μ_t)

    return

end

function run_simulation_SaveActionVecs(r::Array{Float32,3}, v::Array{Float32,3}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any})
    """
    This function runs a single realization of multimodal schooling, using the new vectorized implementation across individuals.
    """

    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]
    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

    # create the shift matrix
    ns_x, ndo_x = gm_params[:ns_x], gm_params[:ndo_x]
    # D_shift::Array{Float32,2} = diagm(ns_x => ones(Float32,ndo_x*ns_x- ns_x));

    # create rotation matrices
    R_starts, R_ends = GeoUtils.generate_start_end_rotation_lists(gp_params[:sector_angles], reverse_flag = true)
    R_list = [R_starts, R_ends]

    noise_struct, dist_matrix, φ_hist, μ_hist, φ_t, dh_dr_self_array, μ_t, ∂F∂v_hist = initialize_history_arrays_SaveActionVecs(gp_params, gm_params)

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        run_single_timestep_SaveActionVecs(r, v, t, dist_matrix, noise_struct, N, D, gp_params,
                                gm_params, R_list, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, ∂F∂v_hist)

    end # end of loop over time

    results_dict = Dict(:r => r, :v => v, :φ_hist => φ_hist, :μ_hist => μ_hist, :∂F∂v_hist => ∂F∂v_hist)

    return results_dict

end # end of run_simulation function

function run_single_timestep_SaveActionVecs(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
        gm_params::Dict{Symbol,Any}, R_starts_ends, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, ∂F∂v_hist)
    """
    Wrapper for all the processes that happen within a single timestep of the overall active inference loop.
        This is the new vectorized version of the schooling function
    """

    # unpack global constant parameters from `gp_params`

    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

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

    r_t, v_past = update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    sector_idx = GeoUtils.assign_visual_field_ids_new(R_starts_ends[1], R_starts_ends[2], r_t, v_past, dist_thr, dist_matrix)

    all_sector_h, all_sector_h_prime, all_dh_dr_self = GeoUtils.calculate_sector_hidden_states_vectorized(dist_matrix, sector_idx, r_t, v_past)

    # x_t = permutedims(cat(hcat(all_sector_h...), hcat(all_sector_h_prime...), dims = 3), (1, 3, 2))

    ####### SAMPLE OBSERVATIONS FROM GENERATIVE PROCESS #######
    noise_samples_φ = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))

    φ_t = get_observations_vectorized(all_sector_h, all_sector_h_prime, noise_samples_φ, gp_params) # N.B. Need to change φ_t that is passed into the run_single_timestep_new function at the top so that it's alsways in vectorized form (not in ns_φ X ndo_φ X N )

    # nullify the empty sectors
    empty_sector_mask = vcat([ hcat(all_sector_h...), hcat(all_sector_h_prime...)]...) .== 0f0
    φ_t[empty_sector_mask] .= 0f0

    ####### UPDATE BELIEFS USING PREDICTIVE CODING #######
    if t == 2
        μ_t[1:(ns_φ*ndo_φ),:] =  φ_t # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
    else
        μ_t = copy(μ_hist[:,:,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
    end

    # μ_t, ε_z = run_belief_updating_vectorized(μ_t, φ_t, gm_params, gp_params, D_shift)
    μ_t, ε_z = run_belief_updating_vectorized(μ_t, φ_t, gm_params, gp_params)

    ####### UPDATE ACTIONS USING ACTIVE INFERENCE  #######
    ∂F∂v = update_action_new(ε_z, gm_params, all_sector_h_prime, all_dh_dr_self)

    v[:,:,t] -= (κ_a .* ∂F∂v)

    v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities
    # x_hist[:,:,:,t] .= copy(x_t)
    φ_hist[:,:,t] .= copy(φ_t)
    μ_hist[:,:,t] .= copy(μ_t)
    ∂F∂v_hist[:,:,t] .= copy(∂F∂v)

    return

end


function update_action_new(ε_z::Array{Float32,2}, gm_params::Dict{Symbol,Any}, all_sector_h_prime, all_dh_dr_self)
    """
    Vectorized implementation of computing the gradients of free energy with respect to actions, computed across individuals
    """

    ∂F∂φprime = ε_z[(gm_params[:ns_φ]+1):(gm_params[:ns_φ]*gm_params[:ndo_φ]),:] # gradient of FE w.r.t to observations
    # ∂φprime_∂v = map((x,y)-> x .* y,  gm_params[:sensory_func_prime].(all_sector_h_prime), all_dh_dr_self) # gradient of observations with respect to actions
    ∂φprime_∂v = map((x,y)-> x .* y,  all_sector_h_prime, all_dh_dr_self) # gradient of observations with respect to actions
    ∂F∂v = hcat(map( (x,y) -> GeoUtils.nandot(x,y), collect(eachcol(∂F∂φprime)), ∂φprime_∂v)...) # gradient of free energy with respect to actions

    return ∂F∂v

end


function run_simulation_new_Perturb(r::Array{Float32,3}, v::Array{Float32,3}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any}, perturb_params::Dict{Symbol,Any})
    """
    This function runs a single realization of multimodal schooling, using the new vectorized implementation across individuals.
    """

    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]
    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

    # create the shift matrix
    ns_x, ndo_x = gm_params[:ns_x], gm_params[:ndo_x]
    D_shift::Array{Float32,2} = diagm(ns_x => ones(Float32,ndo_x*ns_x- ns_x));

    # create rotation matrices
    R_starts, R_ends = GeoUtils.generate_start_end_rotation_lists(gp_params[:sector_angles], reverse_flag = true)
    R_list = [R_starts, R_ends]

    noise_struct, dist_matrix, φ_hist, μ_hist, φ_t, dh_dr_self_array, empty_sector_flags, μ_t = initialize_history_arrays_new(gp_params, gm_params)

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        run_single_timestep_new_Perturb(r, v, t, dist_matrix, noise_struct, N, D, gp_params,
                                gm_params, R_list, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags, perturb_params)

    end # end of loop over time

    results_dict = Dict(:r => r, :v => v, :φ_hist => φ_hist, :μ_hist => μ_hist)

    return results_dict

end # end of run_simulation function


function run_single_timestep_new_Perturb(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
        gm_params::Dict{Symbol,Any}, R_starts_ends, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags, perturb_params::Dict{Symbol,Any})
    """
    Wrapper for all the processes that happen within a single timestep of the overall active inference loop.
        This is the new vectorized version of the schooling function
    """

    # unpack global constant parameters from `gp_params`

    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

    # unpack state space dimensionalities from `gm_params`

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    ns_φ = gm_params[:ns_φ]
    ndo_φ = gm_params[:ndo_φ]

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

    r_t, v_past = update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    sector_idx = GeoUtils.assign_visual_field_ids_new(R_starts_ends[1], R_starts_ends[2], r_t, v_past, dist_thr, dist_matrix)

    all_sector_h, all_sector_h_prime, all_dh_dr_self = GeoUtils.calculate_sector_hidden_states_vectorized(dist_matrix, sector_idx, r_t, v_past)

    ####### SAMPLE OBSERVATIONS FROM GENERATIVE PROCESS #######
    noise_samples_φ = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))

    φ_t = get_observations_vectorized(all_sector_h, all_sector_h_prime, noise_samples_φ, gp_params) # N.B. Need to change φ_t that is passed into the run_single_timestep_new function at the top so that it's alsways in vectorized form (not in ns_φ X ndo_φ X N )

    # nullify the empty sectors
    empty_sector_mask = vcat([ hcat(all_sector_h...), hcat(all_sector_h_prime...)]...) .== 0f0
    φ_t[empty_sector_mask] .= 0f0

    if t in perturb_params[:times]

        # the individuals we perturb are at a set of fixed angles from the school
        perturbed_idx = GeoUtils.id_particles_at_angles(r[:,:,t],v[:,:,t], perturb_params[:perturb_angles])

        sectors_to_bump = perturb_params[:sectors_to_bump] .+ gm_params[:ns_φ] # this moves them up a generalised order
        φ_t[sectors_to_bump,perturbed_idx] = (φ_t[sectors_to_bump,perturbed_idx] .+ perturb_params[:velocity_bump])

    end

    ####### UPDATE BELIEFS USING PREDICTIVE CODING #######
    if t == 2
        μ_t[1:(ns_φ*ndo_φ),:] =  φ_t # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
    else
        μ_t = copy(μ_hist[:,:,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
    end

    μ_t, ε_z = run_belief_updating_vectorized(μ_t, φ_t, gm_params, gp_params)

    ####### UPDATE ACTIONS USING ACTIVE INFERENCE  #######
    ∂F∂v = update_action_new(ε_z, gm_params, all_sector_h_prime, all_dh_dr_self)

    v[:,:,t] -= (κ_a .* ∂F∂v)

    v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities
    # x_hist[:,:,:,t] .= copy(x_t)
    φ_hist[:,:,t] .= copy(φ_t)
    μ_hist[:,:,t] .= copy(μ_t)

    return

end

function perturb_realisations_new(results_dict::Dict{Symbol,Array{Float32, 3}}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any}, perturb_params::Dict{Symbol,Any};
    num_realisations::Int64 = 1, start_idx::Int64 = 10, realisation_length::Int64 = 50, perturb_type::String = "rotation")
    """
    This function runs multiple parallel realisations of a schooling simulation starting a from fixed point in time (fixed beliefs, positions, velocities, etc.)
    with independent noise fluctuations in the dynamics (i.e. independent across realizations). At the starting time point of each realisation, a fixed number of individuals
    (determined by the length of `perturb_params[:perturb_angles]`) are either flipped 90 degrees relative to the motion direction of the school, or are given 'fictive' velocity
    observations according to parameters stored in entries of `perturb_params` (e.g. `perturb_params[:sectors_to_bump]` determines which visual field sectors get fictive observations).

    In case there perturb_type == "rotation", the 90-degree-rotated individuals are chosen to be those at the front of the school (defined as those with the highest projection value
    along the axis parallel to the group's heading vector). Each realisation continues for `realisation_length` timesteps into the future, following the perturbation.

    ARGUMENTS:
    `results_dict`       - [Dict] : stores the history of agent-specific beliefs, positions, velocities, observations, and states for all the individuals
    `gm_params`          - [Dict]: dictionary storing global constant parameters related to the generative models of all agents
    `gp_params`          - [Dict]: dictionary storing global constant parameters related to the generative process of all agents
    `perturb_params`     - [Dict]: dictionary storing parameters related to type (strengh and timing) of perturbation
    `num_realisations`   - [Int64]: number of realisations of the process to run forward
    `start_idx`          - [Int64]: starting point in absolute time (ranging from `1` to `gp_params[:T_sim])` to initialize the perturbations
    `realisation_length` - [Int64]: length of time to simulate each perturbed realisation, in timesteps (where each timestep has duration `gp_params[:dt]`)
    `perturb_type`       - [String]: the type of perturbation, options are either "obs_perturb" or "rotation". In the case that `perturb_type` == "obs_perturb", then the provided
                                    `perturb_params` are used to perturb the velocity observations of particular individuals at particular times.
                                    In case `perturb_type` == "rotation", then we simply rotate a fixed number of individuals at the front of the school
                                    by 90 degrees at the first timestep of the simulation
    """

    # unpack global constant parameters from `gp_params`
    N = gp_params[:N]
    D = gp_params[:D]
    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

    # create the shift matrix
    ns_x, ndo_x = gm_params[:ns_x], gm_params[:ndo_x]

    # create rotation matrices
    R_starts, R_ends = GeoUtils.generate_start_end_rotation_lists(gp_params[:sector_angles], reverse_flag = true)
    R_list = [R_starts, R_ends]

    # pre-allocate arrays to store history of agent-specific variables (e.g. position, velocity, and beliefs), across time and across realisation

    initial_r = copy(results_dict[:r][:,:,start_idx-1])
    initial_v = copy(results_dict[:v][:,:,start_idx-1])
    initial_μ = copy(results_dict[:μ_hist][:,:,start_idx-1])
    initial_φ = copy(results_dict[:φ_hist][:,:,start_idx-1])

    # choose the perturbed individuals based on their ranking in `front-to-backness` along the axis of motion of the school (`d_group`)
    if perturb_type == "rotation"
        num_to_perturb = length(perturb_params[:perturb_angles])
        d_group, _, _, relative_rankings = AnalysisUtils.compute_Dgroup_and_rankings_single(initial_r,initial_v)
        perturbed_idx = convert(Array{Int64,1},relative_rankings[1:num_to_perturb])

        # you fix random individual's heading to vector normal to the group's heading direction
        initial_v[:,perturbed_idx] .= [-d_group[2]; d_group[1]] # this fixes the headings of the perturbed agents to a vector perpendicular to the heading direction
    else
        perturbed_idx = nothing
    end

    noise_struct, dist_matrix, r_all, v_all, μ_all, φ_all = initialise_history_arrays_for_perturbations_new(gp_params, gm_params, num_realisations, realisation_length)
    r_all[:,:,1,:] .= initial_r
    v_all[:,:,1,:] .= initial_v
    μ_all[:,:,1,:] .= initial_μ
    φ_all[:,:,1,:] .= initial_φ

    for real_i = 1:num_realisations

        # create pointer variables for realisation-specific arrays (using @views)
        @views r, v, dist_matrix_iter_i  = r_all[:,:,:,real_i], v_all[:,:,:,real_i], dist_matrix[:,:,real_i]
        @views μ_hist, φ_hist =  μ_all[:,:,:,real_i], φ_all[:,:,:,real_i]

        dh_dr_self_array, φ_t, empty_sector_flags, μ_t = initialise_realisation_cache_new(gp_params, gm_params)

        for t = 2:(realisation_length+1) # we start at 2nd timestep since we've already initialised the first timestep

            run_single_timestep_new_Perturb_v2(r, v, t, dist_matrix_iter_i, noise_struct, N, D, gp_params,
                                    gm_params, R_list, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags, perturb_params, perturbed_idx = perturbed_idx, perturb_type = perturb_type)

        end # end of loop over time
    end


    results_dict = Dict(:r_all => r_all, :v_all => v_all, :μ_all => μ_all, :φ_all => φ_all)

    return results_dict
end

function run_single_timestep_new_Perturb_v2(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
        gm_params::Dict{Symbol,Any}, R_starts_ends, dh_dr_self_array, φ_hist, φ_t, μ_hist, μ_t, empty_sector_flags, perturb_params::Dict{Symbol,Any}; perturbed_idx = nothing, perturb_type::String = "obs_perturb")
    """
    Wrapper for all the processes that happen within a single timestep of the overall active inference loop.
        This is the new vectorized version of the schooling function
    """

    # unpack global constant parameters from `gp_params`

    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

    # unpack state space dimensionalities from `gm_params`

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    ns_φ = gm_params[:ns_φ]
    ndo_φ = gm_params[:ndo_φ]

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

    r_t, v_past = update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    sector_idx = GeoUtils.assign_visual_field_ids_new(R_starts_ends[1], R_starts_ends[2], r_t, v_past, dist_thr, dist_matrix)

    all_sector_h, all_sector_h_prime, all_dh_dr_self = GeoUtils.calculate_sector_hidden_states_vectorized(dist_matrix, sector_idx, r_t, v_past)

    ####### SAMPLE OBSERVATIONS FROM GENERATIVE PROCESS #######
    noise_samples_φ = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))

    φ_t = get_observations_vectorized(all_sector_h, all_sector_h_prime, noise_samples_φ, gp_params) # N.B. Need to change φ_t that is passed into the run_single_timestep_new function at the top so that it's alsways in vectorized form (not in ns_φ X ndo_φ X N )

    # nullify the empty sectors
    empty_sector_mask = vcat([ hcat(all_sector_h...), hcat(all_sector_h_prime...)]...) .== 0f0
    φ_t[empty_sector_mask] .= 0f0

    if t in perturb_params[:times]

        if perturb_type == "obs_perturb"
            # the individuals we perturb are at a set of fixed angles from the school
            perturbed_idx = GeoUtils.id_particles_at_angles(r[:,:,t],v[:,:,t], perturb_params[:perturb_angles])

            sectors_to_bump = perturb_params[:sectors_to_bump] .+ gm_params[:ns_φ] # this moves them up a generalised order
            φ_t[sectors_to_bump,perturbed_idx] = (φ_t[sectors_to_bump,perturbed_idx] .+ perturb_params[:velocity_bump])
        elseif perturb_type == "rotation"
            v[:,perturbed_idx,t] = copy(v[:,perturbed_idx,t-1]) # if we're within stimulation interval, just copy the rotated velocity from the past
        end

    end

    ####### UPDATE BELIEFS USING PREDICTIVE CODING #######
    if t == 2
        μ_t[1:(ns_φ*ndo_φ),:] = φ_t # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
    else
        μ_t = copy(μ_hist[:,:,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
    end

    μ_t, ε_z = run_belief_updating_vectorized(μ_t, φ_t, gm_params, gp_params)

    ####### UPDATE ACTIONS USING ACTIVE INFERENCE  #######
    ∂F∂v = update_action_new(ε_z, gm_params, all_sector_h_prime, all_dh_dr_self)

    if (t in perturb_params[:times]) & (perturb_type == "rotation")
        non_perturbed_idx = setdiff(1:N, perturbed_idx)
        v[:,non_perturbed_idx,t] -= (κ_a .* ∂F∂v[:,non_perturbed_idx])
    else
        v[:,:,t] -= (κ_a .* ∂F∂v)
    end

    v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities
    φ_hist[:,:,t] .= copy(φ_t)
    μ_hist[:,:,t] .= copy(μ_t)

    return

end

function initialise_history_arrays_for_perturbations_new(gp_params::Dict{Symbol, Any}, gm_params::Dict{Symbol, Any}, num_realisations::Int64, realisation_length::Int64)

    # unpack global constant parameters from `gp_params`
    N = gp_params[:N]
    D = gp_params[:D]

    # unpack state space dimensionalities from `gm_params`
    ns_x = gm_params[:ns_x] # number of hidden states
    ndo_x = gm_params[:ndo_x] # generalised orders of hidden states
    ns_φ = gm_params[:ns_φ]  # number of observation dimensions
    ndo_φ = gm_params[:ndo_φ] # generalised orders of observation dimensions

    # create noise structure
    noise_struct = NoiseUtils.NoiseStruct(num_samples = (N*D*ns_φ*ndo_φ*N*num_realisations*(realisation_length+1)) )

    # create distance matrix for storing instantaneous distances between agents
    dist_matrix::Array{Float32,3} = Array{Float32}(undef, N, N, num_realisations)

    r_all::Array{Float32,4}= Array{Float32}(undef, D, N, realisation_length+1, num_realisations) # history of positions

    v_all::Array{Float32,4} = Array{Float32}(undef, D, N, realisation_length+1, num_realisations) # history of velocities

    μ_all::Array{Float32,4} = Array{Float32}(undef, (ns_x*ndo_x), N, realisation_length+1, num_realisations) # history of posterior beliefs
    φ_all::Array{Float32,4} = Array{Float32}(undef, (ns_φ*ndo_φ), N, realisation_length+1, num_realisations) # history of observations

    return noise_struct, dist_matrix, r_all, v_all, μ_all, φ_all

end

function initialise_realisation_cache_new(gp_params::Dict{Symbol,Any}, gm_params::Dict{Symbol,Any})

    N, D = gp_params[:N], gp_params[:D]

    # unpack state space dimensionalities from `gm_params`
    ns_x = gm_params[:ns_x] # number of hidden states
    ndo_x = gm_params[:ndo_x] # generalised orders of hidden states
    ns_φ = gm_params[:ns_φ]  # number of observation dimensions
    ndo_φ = gm_params[:ndo_φ] # generalised orders of observation dimensions

    dh_dr_self_array::Array{Float32,3} = zeros(Float32, D, ns_x, N); # running cache for partial derivatives of hidden states with respect to position of focal particle

    # pre-allocate cache arrays for storing states and observations at a given time step
    φ_t::Array{Float32,2} = zeros(Float32, ns_φ*ndo_φ, N); # running cache for observations

    empty_sector_flags::BitArray{3} = falses(ns_x, ndo_x, N); # running cache for flags that indicate absence of observations for each modality & dyamical order & individual
    μ_t::Array{Float32,2} = zeros(Float32,ns_x*ndo_x, N) # running cache for storing instantaneous beliefs for all individuals

    return dh_dr_self_array, φ_t, empty_sector_flags, μ_t

end


function run_simulation_old(r::Array{Float32,3}, v::Array{Float32,3}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any};
    save_dFdv::Bool=false)
    """
    This function runs a single realization of multimodal schooling.
    """

    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]
    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

    # create the shift matrix
    ns_x, ndo_x = gm_params[:ns_x], gm_params[:ndo_x]
    D_shift::Array{Float32,2} = diagm(ns_x => ones(Float32,ndo_x*ns_x- ns_x));

    # # create rotation matrices
    rotation_matrices = generate_rotation_matrices(gp_params[:sector_angles])

    noise_struct, dist_matrix, x_hist, φ_hist, μ_x, dF_dv_hist, x_t, φ_t, dh_dr_self_array, empty_sector_flags, μ_x_n_t = initialize_history_arrays(gp_params, gm_params)

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        run_single_timestep_old(r, v, t, dist_matrix, noise_struct, N, D, gp_params,
                                gm_params, D_shift, rotation_matrices, dh_dr_self_array,
                                x_hist, x_t, φ_hist, φ_t, μ_x, μ_x_n_t, empty_sector_flags, dF_dv_hist, save_dFdv = save_dFdv)

    end # end of loop over time

    results_dict = Dict(:r => r, :v => v, :x_hist => x_hist, :φ_hist => φ_hist, :μ_x => μ_x, :dF_dv_hist => dF_dv_hist)

    return results_dict

end # end of run_simulation function

function initialize_history_arrays(gp_params::Dict{Symbol, Any}, gm_params::Dict{Symbol, Any})

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

    x_hist = zeros(Float32, ns_x, ndo_x-1, N, T_sim) # history of hidden states
    φ_hist = zeros(Float32, ns_φ,ndo_φ, N, T_sim) # history of observations OLD VERSION

    μ_hist    = zeros(Float32,(ns_x*ndo_x), N, T_sim) # history of beliefs
    dF_dv_hist = zeros(Float32,D, ns_φ, N, T_sim) # history of partial derivatives of F w.r.t to action, divided up by sector

    x_t = zeros(Float32, ns_x, ndo_x-1, N); # running cache for hidden states (used to generate observations)
    φ_t = zeros(Float32, ns_φ,ndo_φ, N); # running cache for observations OLD VERSION

    dh_dr_self_array = zeros(Float32, D, ns_x, N); # running cache for partial derivatives of hidden states with respect to position of focal particle
    empty_sector_flags = falses(ns_x, ndo_x, N); # running cache for flags that indicate absence of observations for each modality & dyamical order & individual

    μ_t = zeros(Float32,ns_x*ndo_x) # running cache for storing instantaneous beliefs for a given individual

    return noise_struct, dist_matrix, x_hist, φ_hist, μ_hist, dF_dv_hist, x_t, φ_t, dh_dr_self_array, empty_sector_flags, μ_t

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

function initialize_history_arrays_SaveActionVecs(gp_params::Dict{Symbol, Any}, gm_params::Dict{Symbol, Any})
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

    ∂F∂v_hist = zeros(Float32, D, N, T_sim) # history of optimal actions (vectors pointing down VFE gradients)

    φ_t = zeros(Float32, (ns_φ*ndo_φ), N); # running cache for observations

    dh_dr_self_array = zeros(Float32, D, ns_x, N); # running cache for partial derivatives of hidden states with respect to position of focal particle

    μ_t = zeros(Float32,ns_x*ndo_x, N) # running cache for storing instantaneous beliefs for a given individual

    return noise_struct, dist_matrix, φ_hist, μ_hist, φ_t, dh_dr_self_array, μ_t, ∂F∂v_hist

end


function update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    # update position using last timestep's velocity
    r[:,:,t] = r[:,:,t-1] .+ (dt .* v[:,:,t-1]) + (z_action .* noise_samples_action);
    r_t = @view r[:,:,t];

    v_past = @view v[:,:,t-1];
    v[:,:,t] = copy(v[:,:,t-1]); # carry forward last timestep's velocity to the current timestep

    # calculate Euclidean distance between all particles
    pairwise!(dist_matrix, Euclidean(), r_t, dims=2); # this goes way faster than the above approach

    return r_t, v_past

end

function run_single_timestep_old(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
    gm_params::Dict{Symbol,Any}, D_shift::Matrix{Float32}, rotation_matrices, dh_dr_self_array, x_hist, x_t, φ_hist, φ_t, μ_x, μ_x_n_t, empty_sector_flags, dF_dv_hist; save_dFdv = false)
    """
    Wrapper for all the processes that happen within a single timestep of the overall active inference loop
    """

    # unpack global constant parameters from `gp_params`

    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

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

    r_t, v_past = update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    noise_samples_φ = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))

    for n = 1:N # loop over individuals

        ####### get hidden states and observations #######

        neighbor_idx = findall(1:N .!== n) # all the particles besides me

        @views my_r, my_v = r_t[:,n], v_past[:,n] # focal particle's position and velocity
        @views other_r, other_v  = r_t[:,neighbor_idx], v_past[:,neighbor_idx] # all other particles' positions and velocities

        sector_idx = GeoUtils.assign_visfield_ids(my_r, my_v, other_r, ns_x, rotation_matrices, dist_thr) # returns a vector of BitArrays, one for each visual sector, that index whether neighbor particles are in the corresponding visual sector

        # calculate hidden states
        x_t[:,:,n], dh_dr_self_array[:,:,n], empty_sector_flags[:,:,n] =  calculate_sector_hidden_states(ns_x, ndo_x, D, sector_idx, dist_matrix[n,:], neighbor_idx, other_r, other_v, my_r, my_v, x_hist[:,:,n,t-1])

        # generate observations from hidden states
        @views hidden_states_n, noise_n = x_t[:,:,n], noise_samples_φ[:,:,n]
        φ_t[:,:,n] = get_observations(ns_φ, ndo_φ, g, ∂g∂x, hidden_states_n, z_gp, noise_n)

        ####### NOTE!!! about converting from matrix [ns_x, ndo_x] representation to vectorised [ns_x * ndo_x,] representation: ##############
        # we flatten arrays in the [ns_x, ndo_x] format into a single long vector, wherehidden state dimensions within the first dynamical order are listed first,
        # before the next dynamical order starts. For example: if a hidden state dimensionality of 5, and 2 dynamical orders: then entries
        # 1 - 5 of the vectorised representation will represent all the states for the first dynamical order, and entries 6 - 10 will store the states for the second dynamical order, etc.
        # We do it this way arbitrarily, but based on the fact that the precision matrices are stored this way (smaller block diagonals represent single dynamical orders, for all hidden states,
        # whereas moving from one block diagonal to the next corresponds to changing dynamical orders).
        ######################################################################################################################################

        vectorized_φ = vec(φ_t[:,:,n]);

        ####### Update beliefs using generalised predictive coding #######

        if t == 2
            μ_x_n_t[1:(ns_φ*ndo_φ)] =  vectorized_φ # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
            μ_x_n_t[(ns_φ*ndo_φ +1):end] .= 0f0 # do this to make sure acceleration beliefs from agent n-1 aren't 'shared' to agent n
        else
            μ_x_n_t = copy(μ_x[:,n,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
        end

        μ_x[:,n,t], ε_z = run_belief_updating(μ_x_n_t, vectorized_φ, dt, num_iter, κ_μ,
                                             ns_x, ndo_x, ns_φ, ndo_φ,
                                             g, ∂g∂x, f, ∂f∂x, 𝚷_z, 𝚷_ω, D_shift)

        update_action_old(n, t, v, ε_z, ns_φ, ndo_φ, ∂g∂x, x_t, dh_dr_self_array, dF_dv_hist, κ_a, save_dFdv)


    end # end of loop over individuals

    v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities
    x_hist[:,:,:,t] .= copy(x_t)
    φ_hist[:,:,:,t] .= copy(φ_t)

    return

end

function update_action_old(n, t, v, ε_z, ns_φ, ndo_φ, ∂g∂x, x_t, dh_dr_self_array, dF_dv_hist, κ_a, save_dFdv)

    ####### Update actions using a gradient descent on free energy #######

    # dF/dv = dF/dphi * dphi/dv -- only non-zero term in the vector
    # dphi/dv = dphi'/dv -- i.e. the h-velocity observation (h_dot). So we only
    # need to use the prediction error related to the observation of
    # the h-velocity

    ∂F∂φprime = ε_z[(ns_φ+1):(ns_φ*ndo_φ)]; # the second ns_φ elements of the vector of sensory prediction errors correspond to the different elements of ∂F∂φprime

    ∂φprime_∂v = ∂g∂x(x_t[:,2,n])' .* dh_dr_self_array[:,:,n];

    if save_dFdv

        for ns_i = 1:ns_φ
            dF_dv_hist[:,ns_i,n,t] = ∂F∂φprime[ns_i] .* ∂φprime_∂v[:,ns_i]; # store the weighted partial derivative for this sector
        end

        ∂F∂v = vec(sum(dF_dv_hist[:,:,n,t],dims=2)) # final free energy gradient is the sum of all the partial derivatives with respect to observations in each visual sector

    else

        ∂F∂v = ∂φprime_∂v*∂F∂φprime; # if you're not keeping track of the individual partial derivatives, then you can do it all in one dot product, like this

    end


    v[:,n,t] .-= (κ_a .* ∂F∂v); # update velocity


end

function run_single_timestep_old_for_perturbations(r, v, t, dist_matrix, noise_struct, N::Int64, D::Int64, gp_params::Dict{Symbol,Any},
    gm_params::Dict{Symbol,Any}, D_shift::Matrix{Float32}, rotation_matrices, dh_dr_self_array, x_hist, x_t, φ_hist, φ_t, μ_x, μ_x_n_t, empty_sector_flags, perturbed_idx; sustain_duration::Int64 = 1)
    """
    Wrapper for all the processes that happen within a single timestep of the overall active inference loop
    """

    # unpack global constant parameters from `gp_params`

    dt = gp_params[:dt]
    z_action = gp_params[:z_action]
    z_gp = gp_params[:z_gp]
    dist_thr = gp_params[:dist_thr]

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

    r_t, v_past = update_positions_velocities_distances(r, v, t, dt, dist_matrix, z_action, noise_samples_action)

    noise_samples_φ = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))

    for n = 1:N # loop over individuals

        ####### get hidden states and observations #######

        neighbor_idx = findall(1:N .!== n) # all the particles besides me

        @views my_r, my_v = r_t[:,n], v_past[:,n] # focal particle's position and velocity
        @views other_r, other_v  = r_t[:,neighbor_idx], v_past[:,neighbor_idx] # all other particles' positions and velocities

        sector_idx = GeoUtils.assign_visfield_ids(my_r, my_v, other_r, ns_x, rotation_matrices, dist_thr) # returns a vector of BitArrays, one for each visual sector, that index whether neighbor particles are in the corresponding visual sector

        # calculate hidden states
        x_t[:,:,n], dh_dr_self_array[:,:,n], empty_sector_flags[:,:,n] =  calculate_sector_hidden_states(ns_x, ndo_x, D, sector_idx, dist_matrix[n,:], neighbor_idx, other_r, other_v, my_r, my_v, x_hist[:,:,n,t-1])

        # generate observations from hidden states
        @views hidden_states_n, noise_n = x_t[:,:,n], noise_samples_φ[:,:,n]
        φ_t[:,:,n] = get_observations(ns_φ, ndo_φ, g, ∂g∂x, hidden_states_n, z_gp, noise_n)

        ####### NOTE!!! about converting from matrix [ns_x, ndo_x] representation to vectorised [ns_x * ndo_x,] representation: ##############
        # we flatten arrays in the [ns_x, ndo_x] format into a single long vector, wherehidden state dimensions within the first dynamical order are listed first,
        # before the next dynamical order starts. For example: if a hidden state dimensionality of 5, and 2 dynamical orders: then entries
        # 1 - 5 of the vectorised representation will represent all the states for the first dynamical order, and entries 6 - 10 will store the states for the second dynamical order, etc.
        # We do it this way arbitrarily, but based on the fact that the precision matrices are stored this way (smaller block diagonals represent single dynamical orders, for all hidden states,
        # whereas moving from one block diagonal to the next corresponds to changing dynamical orders).
        ######################################################################################################################################

        vectorized_φ = vec(φ_t[:,:,n]);

        ####### Update beliefs using generalised predictive coding #######

        if t == 2
            μ_x_n_t[1:(ns_φ*ndo_φ)] =  vectorized_φ # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
            μ_x_n_t[(ns_φ*ndo_φ +1):end] .= 0f0 # do this to make sure acceleration beliefs from agent n-1 aren't 'shared' to agent n
        else
            μ_x_n_t = copy(μ_x[:,n,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
        end

        μ_x[:,n,t], ε_z = run_belief_updating(μ_x_n_t, vectorized_φ, dt, num_iter, κ_μ,
                                             ns_x, ndo_x, ns_φ, ndo_φ,
                                             g, ∂g∂x, f, ∂f∂x, 𝚷_z, 𝚷_ω, D_shift)

        update_action_old_for_perturbations(n, t, v, ε_z, ns_φ, ndo_φ, ∂g∂x, x_t, dh_dr_self_array, κ_a)

    end # end of loop over individuals

    v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities

    if t <= sustain_duration
        v[:,perturbed_idx,t] = copy(v[:,perturbed_idx,t-1]) # if we're within the sustain duration, just copy velocity from the past
    end

    x_hist[:,:,:,t] .= copy(x_t)
    φ_hist[:,:,:,t] .= copy(φ_t)

    return

end

function update_action_old_for_perturbations(n, t, v, ε_z, ns_φ, ndo_φ, ∂g∂x, x_t, dh_dr_self_array, κ_a)

    ∂F∂φprime = ε_z[(ns_φ+1):(ns_φ*ndo_φ)]; # the second ns_φ elements of the vector of sensory prediction errors correspond to the different elements of ∂F∂φprime

    ∂φprime_∂v = ∂g∂x(x_t[:,2,n])' .* dh_dr_self_array[:,:,n];

    ∂F∂v = ∂φprime_∂v*∂F∂φprime; # if you're not keeping track of the individual partial derivatives, then you can do it all in one dot product, like this

    v[:,n,t] .-= (κ_a .* ∂F∂v); # update velocity


end


function perturb_realisations(results_dict::Dict{Symbol,Array{Float32}}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any};
    num_realisations::Int64 = 1, num_to_perturb::Int64 = 1, start_idx::Int64 = 10, realisation_length::Int64 = 50, sustain_duration::Int64 = 1)
    """
    This function runs multiple parallel realisations of a schooling simulation starting a from fixed point in time (fixed beliefs, positions, velocities, etc.)
    with independent noise fluctuations in the dynamics (i.e. independent across realizations). At the starting time point of each realisation, a fixed number of individuals `num_to_perturb`
    are flipped 90 degrees in heading direction, to simulate an external 'perturbation' to the school. The individuals are chosen to be those at the front of the school
    (defined as those with the highest projection value along the axis parallel to the group's heading vector). Each realisation continues for `realisation_length` timesteps into
    the future. If num_to_perturb == 0, realisations are run forward without any external perturbations - this might serve as the `control` set of realisations for a set of perturbed realisations.
    The rotation can be sustained for a desired number of timesteps, specified in the parameter `sustain_duration`. The individuals to be perturbed are chosen to be those at the front of the school
    at the beginning of the perturbation.

    ARGUMENTS:
    `results_dict` - [Dict] : stores the history of agent-specific beliefs, positions, velocities, observations, and states for all the individuals
    `gm_params` - [Dict]:
    `gp_params` - [Dict]:
    `num_realisations` - [Int64]: number of realisations of the process to run forward
    `num_to_perturb` - [Int64]: number of agents to perturb in a given realisation
    `start_idx` - [Int64]: starting point in time for the perturbations
    `realisation_length` - [Int64]: length of each realisation, in timesteps (where each timestep has duration `gp_params[:dt]`)
    `sustain_duration` - [Int64]: the duration in timesteps to sustain the rotation of the perturbed-individuals' velocities
     """

    # unpack global constant parameters from `gp_params`
    N = gp_params[:N]
    D = gp_params[:D]

    # unpack state space dimensionalities from `gm_params`

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    # create the shift matrix
    D_shift::Array{Float32,2} = diagm(ns_x => ones(Float32,ndo_x*ns_x - ns_x));

    # create rotation matrices
    rotation_matrices = generate_rotation_matrices(gp_params[:sector_angles])

    # pre-allocate arrays to store history of agent-specific variables (e.g. position, velocity, and beliefs), across time and across realisation

    initial_r = copy(results_dict[:r][:,:,start_idx-1])
    initial_v = copy(results_dict[:v][:,:,start_idx-1])
    initial_μ = copy(results_dict[:μ_x][:,:,start_idx-1])
    initial_x = copy(results_dict[:x_hist][:,:,:,start_idx-1])
    initial_φ = copy(results_dict[:φ_hist][:,:,:,start_idx-1])

    # choose the perturbed individuals based on their ranking in `front-to-backness` along the axis of motion of the school (`d_group`)

    if num_to_perturb > 0
        d_group, _, _, relative_rankings = AnalysisUtils.compute_Dgroup_and_rankings_single(initial_r,initial_v)
        perturbed_idx = convert(Array{Int64,1},relative_rankings[1:num_to_perturb])

        # generate rotation matrix
        rotation_matrix::Array{Float32,2} = [0f0 -1f0; 1f0 0f0] # 90 degree rotation matrix

        # Version 1 (what I tried first): you rotate random individual at front of school's vector by 90 degrees
        # initial_v[:,perturbed_idx] = mapslices(x -> rotation_matrix*x,initial_v[:,perturbed_idx], dims=1) # this rotates every agent's velocity by 90 degrees

        # Version 2 (what I tried next, after reading the paper more): you fix random individual's heading to vector normal to the group's heading direction
        initial_v[:,perturbed_idx] .= rotation_matrix*d_group # this fixes the headings of the perturbed agents  to a vector perpendicular to the heading direction

    end

    noise_struct, dist_matrix, r_all, v_all, μ_all, x_all, φ_all = initialise_history_arrays_for_perturbations(gp_params, gm_params, num_realisations, realisation_length)
    r_all[:,:,1,:] .= initial_r
    v_all[:,:,1,:] .= initial_v
    μ_all[:,:,1,:] .= initial_μ
    x_all[:,:,:,1,:] .= initial_x
    φ_all[:,:,:,1,:] .= initial_φ


    for iter_i = 1:num_realisations

        # create pointer variables for realisation-specific arrays (using @views)
        @views r, v, dist_matrix_iter_i  = r_all[:,:,:,iter_i], v_all[:,:,:,iter_i], dist_matrix[:,:,iter_i]
        @views μ_x, x_hist, φ_hist =  μ_all[:,:,:,iter_i], x_all[:,:,:,:,iter_i], φ_all[:,:,:,:,iter_i]

        x_t, φ_t, dh_dr_self_array, empty_sector_flags, μ_x_n_t = initialise_realisation_cache(gp_params, gm_params)

        for t = 2:(realisation_length+1) # we start at 2nd timestep since we've already initialised the first timestep

            run_single_timestep_old_for_perturbations(r, v, t, dist_matrix_iter_i, noise_struct, N, D, gp_params,
                                    gm_params, D_shift, rotation_matrices, dh_dr_self_array,
                                    x_hist, x_t, φ_hist, φ_t, μ_x, μ_x_n_t, empty_sector_flags, perturbed_idx, sustain_duration = sustain_duration)

        end # end of loop over time

    end # end of loop over parallel realisations

    results_dict = Dict(:r_all => r_all, :v_all => v_all, :x_all => x_all, :φ_all => φ_all, :μ_all => μ_all)

    return results_dict

end # end of run_simulation function

function initialise_history_arrays_for_perturbations(gp_params::Dict{Symbol, Any}, gm_params::Dict{Symbol, Any}, num_realisations::Int64, realisation_length::Int64)

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
    noise_struct = NoiseUtils.NoiseStruct(num_samples = (N*D*ns_φ*ndo_φ*N*num_realisations*(realisation_length+1)) )

    # create distance matrix for storing instantaneous distances between agents
    dist_matrix::Array{Float32,3} = zeros(Float32,N,N,num_realisations)

    r_all::Array{Float32,4}= zeros(Float32, D, N, realisation_length+1, num_realisations) # history of positions

    v_all::Array{Float32,4} = zeros(Float32, D, N, realisation_length+1, num_realisations) # history of velocities

    μ_all::Array{Float32,4} = zeros(Float32,(ns_x*ndo_x), N, realisation_length+1, num_realisations) # history of posterior beliefs

    x_all::Array{Float32,5} = zeros(Float32, ns_x, ndo_x-1, N, realisation_length+1, num_realisations) # history of hidden states

    φ_all::Array{Float32,5} = zeros(Float32, ns_φ, ndo_φ, N, realisation_length+1, num_realisations) # history of observations

    return noise_struct, dist_matrix, r_all, v_all, μ_all, x_all, φ_all

end

function initialise_realisation_cache(gp_params::Dict{Symbol,Any}, gm_params::Dict{Symbol,Any})

    N, D = gp_params[:N], gp_params[:D]

    # unpack state space dimensionalities from `gm_params`
    ns_x = gm_params[:ns_x] # number of hidden states
    ndo_x = gm_params[:ndo_x] # generalised orders of hidden states
    ns_φ = gm_params[:ns_φ]  # number of observation dimensions
    ndo_φ = gm_params[:ndo_φ] # generalised orders of observation dimensions

    # pre-allocate cache arrays for storing states and observations at a given time step
    x_t::Array{Float32,3} = zeros(Float32, ns_x, ndo_x-1, N); # running cache for hidden states (used to generate observations)
    φ_t::Array{Float32,3} = zeros(Float32, ns_φ, ndo_φ, N); # running cache for observations

    dh_dr_self_array::Array{Float32,3} = zeros(Float32, D, ns_x, N); # running cache for partial derivatives of hidden states with respect to position of focal particle
    empty_sector_flags::BitArray{3} = falses(ns_x, ndo_x, N); # running cache for flags that indicate absence of observations for each modality & dyamical order & individual
    μ_x_n_t::Array{Float32,1} = zeros(Float32,ns_x*ndo_x) # running cache for storing instantaneous beliefs for a given individual

    return x_t, φ_t, dh_dr_self_array, empty_sector_flags, μ_x_n_t

end


function initialize_and_run_perturbation(D::Int64, N::Int64, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any}, stimulus_params::Dict{Symbol,Int64})
    """
    This function runs a batch of perturbation trials (sustained stimulus version)
    A simulation is first run until an instance with cohesion throughout is obtained, and then
    a series of `n_realisations` perturbations are run forward from `perturb_idx` timestamp with stimulus duration `sustain_duration`.
    Both the original full simulation results and the perturbation results are returned in dictionary form
    """

    perturb_idx = stimulus_params[:perturb_idx]
    track_duration::Int64 = stimulus_params[:track_duration]
    num_to_perturb::Int64 = stimulus_params[:num_to_perturb]
    sustain_duration::Int64 = stimulus_params[:sustain_duration]
    n_realisations::Int64 = stimulus_params[:n_realisations]

    r, v = initialise_positions_velocities(gp_params[:T_sim], D, N, position_bounds=[-0.5f0,0.5f0])

    results_dict = run_simulation_old(r, v, gm_params, gp_params)

    # make sure the school was fully connected throughout the period of the perturbation
    c_t = AnalysisUtils.is_connected_over_time(results_dict[:r][:,:,perturb_idx:(perturb_idx+track_duration)], threshold = 5f0)

    # if the school is not connected, re-initialise and run the simulation until you get to a run where it _is_ connected.
    while any(c_t .== 0)
        r, v = initialise_positions_velocities(gp_params[:T_sim], D, N, position_bounds=[-0.5f0,0.5f0])
        results_dict = run_simulation_old(r, v, gm_params, gp_params)
        c_t = AnalysisUtils.is_connected_over_time(results_dict[:r][:,:,perturb_idx:(perturb_idx+track_duration)], threshold = 5f0)
    end

    perturbation_results = perturb_realisations(results_dict, gm_params, gp_params,
                            num_realisations = n_realisations, num_to_perturb = num_to_perturb,
                            start_idx = perturb_idx, realisation_length = track_duration, sustain_duration = sustain_duration)

    return results_dict, perturbation_results
end



function generate_default_gp_params(N, T, D; ns_φ = 4, ndo_φ = 2, dt = 0.01f0, dist_thr = 7f0, sector_angles = [120f0, 60f0, 0f0, 360f0 - 60f0, 360f0 - 120f0],
                                    z_gp = 0.1f0, z_action = 0.05f0, α_g = 10f0, b = 3.5f0)

    t_axis = 0:dt:T; # time axis in seconds
    T_sim = length(t_axis); # length of time-axis in number of samples

    EM_scalar = sqrt(dt)
    z_gp = EM_scalar .* z_gp * ones(Float32,ns_φ,ndo_φ)
    z_action = EM_scalar .* (z_action.* ones(Float32,D))

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
    sampling_func_φ(h, noise) = g.(h) .+ z_gp[:,1] .* noise[:,1]
    h_by_hprime(h,hprime) = ∂g∂x.(h) .* hprime
    sampling_func_φprime(h_x_hprime,noise) = h_x_hprime .+ z_gp[:,2] .* noise[:,2]


    gp_params = Dict(:N => N, :D => D, :T_sim => T_sim, :dt => dt, :dist_thr => dist_thr,
                    :sector_angles => sector_angles, :EM_scalar => EM_scalar, :z_gp => z_gp, :z_action => z_action,
                    :sampling_func_φ => sampling_func_φ, :h_by_hprime => h_by_hprime, :sampling_func_φprime => sampling_func_φprime)

    return gp_params
end

function generate_default_gp_params_linear_g(N, T, D; ns_φ = 4, ndo_φ = 2, dt = 0.01f0, dist_thr = 7f0, sector_angles = [120f0, 60f0, 0f0, 360f0 - 60f0, 360f0 - 120f0],
                                    z_gp = 0.1f0, z_action = 0.05f0)

    t_axis = 0:dt:T; # time axis in seconds
    T_sim = length(t_axis); # length of time-axis in number of samples

    EM_scalar = sqrt(dt)
    z_gp = EM_scalar .* z_gp * ones(Float32,ns_φ,ndo_φ)
    z_action = EM_scalar .* (z_action.* ones(Float32,D))

    function g(x)
        return x
    end

    function ∂g∂x(x)
        return 1f0
    end

    # sampling functions of the generative process
    sampling_func_φ(h, noise) = g.(h) .+ z_gp[:,1] .* noise[:,1]
    h_by_hprime(h,hprime) = ∂g∂x.(h) .* hprime
    sampling_func_φprime(h_x_hprime,noise) = h_x_hprime .+ z_gp[:,2] .* noise[:,2]


    gp_params = Dict(:N => N, :D => D, :T_sim => T_sim, :dt => dt, :dist_thr => dist_thr,
                    :sector_angles => sector_angles, :EM_scalar => EM_scalar, :z_gp => z_gp, :z_action => z_action,
                    :sampling_func_φ => sampling_func_φ, :h_by_hprime => h_by_hprime, :sampling_func_φprime => sampling_func_φprime)

    return gp_params
end


function run_belief_updating_vectorized_old(μ::Array{Float32,2}, φ::Array{Float32,2}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any})
    """
    This function runs gradient descent in generalised coordinates for a given timepoint to optimise the beliefs about
    hidden states μ and their higher orders of motion μ̃ in a moving frame of reference.
    ARGUMENTS
    =========
    `μ` - [Array{Float32,2}] - matrix of beliefs about hidden states and their higher orders of motion for each individual, of size (ns_x * ndo_x, N),
                                with hidden state dimensions being stored first/together, with rows of subsequent hidden states
                                representing those same states at higher and higher generalised orders (e.g. velocity, acceleration, etc.)
    `φ` - [Array{Float32,2}] - matrix of observations of hidden states and its observation higher orders of motion for each individual, of size (ns_φ * ndo_φ, N)
    `gm_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative model
    `gp_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative process
    """

    N = gp_params[:N]
    dt = gp_params[:dt]

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    ns_φ = gm_params[:ns_φ]
    ndo_φ = gm_params[:ndo_φ]

    num_iter = gm_params[:num_iter]
    κ_μ = gm_params[:κ_μ]

    g = gm_params[:sensory_func]
    ∂g∂x = gm_params[:sensory_func_prime]

    f = gm_params[:flow_func]
    ∂f∂x = gm_params[:flow_func_prime]

    𝚷_z = gm_params[:𝚷_z]
    𝚷_ω = gm_params[:𝚷_ω]

    D_shift = gm_params[:D_shift]
    D_T = gm_params[:D_T]

    ε_z::Array{Float32,2} = zeros(Float32,ns_x*ndo_x, N) # running cache for storing instantaneous precision-weighted sensory prediction errors
    f_μ::Array{Float32,2} = zeros(Float32, ns_x*ndo_x, N) # running cache for storing instantaneous process expectations

    # new matrix-valued derivatives version
    f_gradients::Array{Float32,2} = zeros(Float32, ns_x*ndo_x, ns_x*ndo_x) # running cache for storing instantaneous process derivatives
    ∂f∂x_μ3::Array{Float32,2} = zeros(Float32, ns_x, ns_x) # f-gradients at third order (stipulatively 0)

    for ii = 1:num_iter

        # sensory and process partial derivatives
        ∂g∂x_μ1 = ∂g∂x(μ[1:ns_x,:]);
        ∂g∂x_μ2 = ∂g∂x(μ[1:ns_x,:]);

        @views g_gradients = [∂g∂x_μ1; ∂g∂x_μ2];

        # where the function ∂f∂x returns a matrix
        ∂f∂x_μ1 = ∂f∂x(μ[1:ns_x,:]);
        ∂f∂x_μ2 = ∂f∂x(μ[1:ns_x,:]);

        # matrix version
        f_gradients = BlockDiagonal([∂f∂x_μ1, ∂f∂x_μ2, ∂f∂x_μ3])

        g_μ = [ g(μ[1:ns_φ,:]); ∂g∂x_μ1 .* μ[(ns_φ+1):(ns_x*ndo_φ),:] ];

        s_pe = φ .- g_μ; # sensory prediction error: observations (φ) minus expectations (g_μ)

        # set sensory prediction errors equal to 0 if they come from empty sectors (empty sectors have exactly `0f0` as observations)
        s_pe[φ .== 0f0] .= 0f0

        p_weighted_spe = 𝚷_z * s_pe;

        # set precision-weighted sensory prediction errors equal to 0 if they come from empty sectors (empty sectors have exactly `0f0` as observations)
        p_weighted_spe[φ .== 0f0] .= 0f0

        ε_z[1:(ns_φ*ndo_φ),:] = g_gradients .* p_weighted_spe;

        # where the function ∂f∂x returns a matrix
        f_μ[1:(ns_x*(ndo_x-1)),:] = [ f(μ[1:ns_x,:]); ∂f∂x_μ1 * μ[(ns_x+1):(ns_x*(ndo_x-1)),:] ]

        p_pe = D_shift*μ .- f_μ; # process prediction errors (between the parallel/same level and the level above)

        p_weighted_ppe = 𝚷_ω * p_pe;

        # new version, where f_gradients is single matrix
        ε_ω = f_gradients * p_weighted_ppe .- D_T'*p_weighted_ppe;

        ∂μ∂t = D_shift*μ .+ ε_z .+ ε_ω;

        μ .+= κ_μ .* ∂μ∂t;

    end

    return μ, ε_z

end

function run_belief_updating_vectorized(μ::Array{Float32,2}, φ::Array{Float32,2}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any})

    """
    This function runs gradient descent in generalised coordinates for a given timepoint to optimise the beliefs about
    hidden states μ and their higher orders of motion μ̃ in a moving frame of reference.
    ARGUMENTS
    =========
    `μ` - [Array{Float32,2}] - matrix of beliefs about hidden states and their higher orders of motion for each individual, of size (ns_x * ndo_x, N),
                                with hidden state dimensions being stored first/together, with rows of subsequent hidden states
                                representing those same states at higher and higher generalised orders (e.g. velocity, acceleration, etc.)
    `φ` - [Array{Float32,2}] - matrix of observations of hidden states and its observation higher orders of motion for each individual, of size (ns_φ * ndo_φ, N)
    `gm_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative model
    `gp_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative process
    """

    N = gp_params[:N]
    dt = gp_params[:dt]

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    ns_φ = gm_params[:ns_φ]
    ndo_φ = gm_params[:ndo_φ]

    num_iter = gm_params[:num_iter]
    κ_μ = gm_params[:κ_μ]

    𝚷_z = gm_params[:𝚷_z]
    𝚷_ω = gm_params[:𝚷_ω]

    # NEW NEW VERSION
    ∇f_T = gm_params[:∇f]
    tilde_f = gm_params[:tilde_f]
    D_shift = gm_params[:D_shift]
    D_T = gm_params[:D_T]
    g = gm_params[:sensory_func]
    ∇g = gm_params[:∇g]

    ε_z::Array{Float32,2} = zeros(Float32,ns_x*ndo_x, N) # running cache for storing instantaneous precision-weighted sensory prediction errors

    for ii = 1:num_iter

        # NEW NEW VERSION
        s_pe = φ .- g(μ); # sensory prediction error: observations (φ) minus expectations (g(μ))

        # set sensory prediction errors equal to 0 if they come from empty sectors (empty sectors have exactly `0f0` as observations)
        s_pe[φ .== 0f0] .= 0f0

        p_weighted_spe = 𝚷_z * s_pe;

        # set precision-weighted sensory prediction errors equal to 0 if they come from empty sectors (empty sectors have exactly `0f0` as observations)
        p_weighted_spe[φ .== 0f0] .= 0f0

        # NEW NEW VERSION
        ε_z = ∇g * p_weighted_spe

        # NEW NEW VERSION
        p_pe = D_shift*μ .- tilde_f(μ)

        p_weighted_ppe = 𝚷_ω * p_pe;

        # NEW NEW VERSION
        ε_ω = ∇f_T * p_weighted_ppe .- D_T*p_weighted_ppe;

        ∂μ∂t = D_shift*μ .+ ε_z .+ ε_ω;

        μ .+= κ_μ .* ∂μ∂t;

    end

    return μ, ε_z

end

function belief_updating_one_target(μ, φ, gm_params, gp_params, D_shift)

    """
    This function runs gradient descent in generalised coordinates for a given timepoint to optimise the beliefs about
    hidden states μ that represent the distance to a target and their higher orders of motion μ̃ in a moving frame of reference.
    ARGUMENTS
    =========
    `μ` - [Array{Float32,2}] - matrix of beliefs about hidden states (about stargets) and their higher orders of motion for each individual, of size (ndo_x, N),
                                with rows of subsequent hidden states indexing higher and higher generalised orders (e.g. velocity, acceleration, etc.)
    `φ` - [Array{Float32,2}] - matrix of observations of hidden states and its observation at higher orders of motion for each individual, of size (ndo_φ, N)
    `gm_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative model
    `gp_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative process
    `D_shift`   [Matrix{Float32, 2}] - shift operator that shifts 'up' a vector of generalised coordinates, so that now (D_shift*mu)[i] = mu[i+1], and mu[end] = 0.0
    """

    N = size(μ,2)
    dt = gp_params[:dt]

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    num_iter = gm_params[:num_iter]
    κ_μ = gm_params[:κ_μ]

    f = gm_params[:flow_func_tdist]
    ∂f∂x = gm_params[:flow_func_tdistprime]

    𝚷_z = gm_params[:𝚷_z_tdist]
    𝚷_ω = gm_params[:𝚷_ω_tdist]

    f_gradients::Array{Float32,2} = zeros(Float32, ndo_x, N) # running cache for storing instantaneous process derivatives

    ε_z::Array{Float32,2} = zeros(Float32, ndo_x, N) # running cache for storing instantaneous precision-weighted sensory prediction errors

    f_μ::Array{Float32,2} = zeros(Float32, ndo_x, N) # running cache for storing instantaneous process expectations

    for ii = 1:num_iter

        # the partial derivatives of the sensory mapping (g) are all gonna be 1.0 in this case
        # since we don't have a nonlinear observation function

        # ∂g∂x_μ1 = ∂g∂x(μ[1:ns_x,:]);
        # ∂g∂x_μ2 = ∂g∂x(μ[1:ns_x,:]);
        #
        # @views g_gradients = [∂g∂x_μ1; ∂g∂x_μ2];

        ∂f∂x_μ1 = ∂f∂x.(μ[1,:]);
        ∂f∂x_μ2 = ∂f∂x.(μ[1,:]);

        f_gradients[1,:] = ∂f∂x_μ1 # gradients at order 0
        f_gradients[2,:] = ∂f∂x_μ2 # gradients at order 1 (velocity)

        # g_μ = [ g(μ[1:ns_φ,:]); ∂g∂x_μ1 .* μ[(ns_φ+1):(ns_x*ndo_φ),:] ];
        # can simplify this since we know g(x) is identity
        # g_μ = [ μ[1,:]';  μ[2,:]'];
        g_μ = copy(μ[1:2,:])

        s_pe = φ .- g_μ; # sensory prediction error: observations (φ) minus expectations (g_μ)

        p_weighted_spe = 𝚷_z * s_pe;

        # ε_z[1:2,:] = g_gradients .* p_weighted_spe;
        ε_z[1:2,:] = p_weighted_spe;

        f_μ[1:2,:] = [ f(μ[1,:])'; (∂f∂x_μ1 .* μ[2,:])' ] # process expectations (the last order is all zeros - no 'same-level' information at highest order)

        p_pe = D_shift*μ .- f_μ; # process prediction errors (between the parallel/same level and the level above)

        p_weighted_ppe = 𝚷_ω * p_pe;

        ε_ω = f_gradients .* p_weighted_ppe .- D_shift'*p_weighted_ppe;

        ∂μ∂t = D_shift*μ .+ ε_z .+ ε_ω;

        μ .+= κ_μ .* ∂μ∂t;

    end

    return μ, ε_z

end

function run_belief_updating_tdist(μ_t_target, φ_t_target, gm_params, gp_params, D_shift_tdist)
    """
    This function runs gradient descent in generalised coordinates for a given timepoint to optimise the beliefs about
    hidden states μ and their higher orders of motion μ̃ in a moving frame of reference, for difference targets (independent elements of μ_t_target)
    ARGUMENTS
    =========
    `μ` - [Vector{Array{Float32,2}}] - vector containing matrices of beliefs about tdist hidden states and their higher orders of motion across individuals, one matrix per target. Each sub-matrix of the vector is
                                of size (ndo_x, N) with rows indexing higher generalised orders (e.g. velocity, acceleration, etc.)
    `φ` - [Vector{Array{Float32,2}}] - vector containing matrices of observations of tdist hidden states and their higher orders of motion across individuals, one matrix per target.
                                Each sub-matrix of the vector is of size (ndo_φ, N) with rows indexing higher generalised orders (e.g. velocity, acceleration, etc.)
    `gm_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative model
    `gp_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative process
    """

    # first get the function of a single target's belief and observation array, and fix the gm_params and gp_params arguments
    belief_update_f = (μ_t, φ_t) -> belief_updating_one_target(μ_t, φ_t, gm_params, gp_params, D_shift_tdist)

    # then map the single-array version of the belief updating function across different targets
    mapped_μ_and_ε = map(belief_update_f, μ_t_target, φ_t_target)

    μ_t_all_targets = map(x -> x[1], mapped_μ_and_ε)
    ε_z_all_targets = map(x -> x[2], mapped_μ_and_ε)

    return μ_t_all_targets, ε_z_all_targets

end

function belief_updating_one_target_withGammaPlearning(μ, φ, Γ_z_params, gm_params, gp_params, D_shift)

    """
    This function runs gradient descent in generalised coordinates for a given timepoint to optimise the beliefs about
    hidden states μ that represent the distance to a target and their higher orders of motion μ̃ in a moving frame of reference.
    ARGUMENTS
    =========
    `μ` - [Array{Float32,2}] - matrix of beliefs about hidden states (about stargets) and their higher orders of motion for each individual, of size (ndo_x, N),
                                with rows of subsequent hidden states indexing higher and higher generalised orders (e.g. velocity, acceleration, etc.)
    `φ` - [Array{Float32,2}] - matrix of observations of hidden states and its observation at higher orders of motion for each individual, of size (ndo_φ, N)
     `Γ_z_params` - [Vector{Array{Float32,2}}] - vector of two matrices: the first contains the α parameters of Γ conjugate prior (posterior) over sensory precision, one for each higher order of motion &  individual.
                                                The second contains the β parameters of the conjugate prior (posterior), again per higher order of motion and per individual. Each matrix Γ_z_params[i] is of size (ndo_φ, N)
    `gm_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative model
    `gp_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative process
    `D_shift`   [Matrix{Float32, 2}] - shift operator that shifts 'up' a vector of generalised coordinates, so that now (D_shift*mu)[i] = mu[i+1], and mu[end] = 0.0
    """

    N = size(μ,2)
    dt = gp_params[:dt]

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    num_iter = gm_params[:num_iter]
    κ_μ = gm_params[:κ_μ]

    f = gm_params[:flow_func_tdist]
    ∂f∂x = gm_params[:flow_func_tdistprime]

    𝚷_ω = gm_params[:𝚷_ω_tdist]

    λ::Float32 = gm_params[:λ]

    f_gradients::Array{Float32,2} = zeros(Float32, ndo_x, N) # running cache for storing instantaneous process derivatives

    ε_z::Array{Float32,2} = zeros(Float32, ndo_x, N) # running cache for storing instantaneous precision-weighted sensory prediction errors

    f_μ::Array{Float32,2} = zeros(Float32, ndo_x, N) # running cache for storing instantaneous process expectations


    for ii = 1:num_iter

        # the partial derivatives of the sensory mapping (g) are all gonna be 1.0 in this case
        # since we don't have a nonlinear observation function

        # ∂g∂x_μ1 = ∂g∂x(μ[1:ns_x,:]);
        # ∂g∂x_μ2 = ∂g∂x(μ[1:ns_x,:]);
        #
        # @views g_gradients = [∂g∂x_μ1; ∂g∂x_μ2];

        ∂f∂x_μ1 = ∂f∂x.(μ[1,:]);
        ∂f∂x_μ2 = ∂f∂x.(μ[1,:]);

        f_gradients[1,:] = ∂f∂x_μ1 # gradients at order 0
        f_gradients[2,:] = ∂f∂x_μ2 # gradients at order 1 (velocity)

        # g_μ = [ g(μ[1:ns_φ,:]); ∂g∂x_μ1 .* μ[(ns_φ+1):(ns_x*ndo_φ),:] ];
        # can simplify this since we know g(x) is identity
        # g_μ = [ μ[1,:]';  μ[2,:]'];
        g_μ = copy(μ[1:2,:])

        s_pe = φ .- g_μ; # sensory prediction error: observations (φ) minus expectations (g_μ)

        # if 𝚷_z is d x d x N, and s_pe is d x 1 x N, then we can use NNLib.batched_mul to multiply each precision matrix with each individual's sensory precision error
        # p_weighted_spe = dropdims(batched_mul(𝚷_z, reshape(s_pe, ndo_x-1, 1, N)), dims = 2)

        # if 𝚷_z is diagonal for all individuals, then all we have to do is multiply each row of s_pe by E[ω] for each variable
        #E[ω] = α / β
        p_weighted_spe = (Γ_z_params[1] ./ Γ_z_params[2]) .* s_pe # expected value of the posterior precision (a gamma distribution), scaling the sensory prediction error

        # ε_z[1:2,:] = g_gradients .* p_weighted_spe;
        ε_z[1:2,:] = p_weighted_spe;

        f_μ[1:2,:] = [ f(μ[1,:])'; (∂f∂x_μ1 .* μ[2,:])' ] # process expectations (the last order is all zeros - no 'same-level' information at highest order)

        p_pe = D_shift*μ .- f_μ; # process prediction errors (between the parallel/same level and the level above)

        p_weighted_ppe = 𝚷_ω * p_pe;

        ε_ω = f_gradients .* p_weighted_ppe .- D_shift'*p_weighted_ppe;

        ∂μ∂t = D_shift*μ .+ ε_z .+ ε_ω;

        μ .+= κ_μ .* ∂μ∂t;

    end

    # fixed form update for Γ_z_params[1] and Γ_z_params[2], with geometric weighting given by λ
    Γ_z_params[1] = (λ .* Γ_z_params[1]) .+ 0.5f0
    Γ_z_params[2] = (λ .* Γ_z_params[2]) .+ (0.5f0 .* (φ .- μ[1:2,:]).^2) # this update uses the latest posterior (that's already been optimized)

    return μ, ε_z, Γ_z_params

end


function run_belief_updating_tdist_withGammaPlearning(μ_t_target, φ_t_target, Γ_z_target, gm_params, gp_params, D_shift_tdist)
    """
    This function runs gradient descent in generalised coordinates for a given timepoint to optimise the beliefs about
    hidden states μ and their higher orders of motion μ̃ in a moving frame of reference, for difference targets (independent elements of μ_t_target).
    We also update the precisions using exact Bayesian updating scheme for gamma conjugate priors described in Baioumy et al. (2022):  "Precision from History: Fault-tolerant Control
    for Sensory Faults via Bayesian Inference and Geometric Weighting"
    ARGUMENTS
    =========
    `μ` - [Vector{Array{Float32,2}}] - vector containing matrices of beliefs about tdist hidden states and their higher orders of motion across individuals, one matrix per target. Each sub-matrix of the vector is
                                of size (ndo_x, N) with rows indexing higher generalised orders (e.g. velocity, acceleration, etc.)
    `φ` - [Vector{Array{Float32,2}}] - vector containing matrices of observations of tdist hidden states and their higher orders of motion across individuals, one matrix per target.
                                Each sub-matrix of the vector is of size (ndo_φ, N) with rows indexing higher generalised orders (e.g. velocity, acceleration, etc.)
    `Γ_z_target` - [Vector{Vector{Array{Float32,2}}}] - vector whose elements contain collections of posterior statistics of target sensory precisions and their higher orders of motion, across individuals. Two vectors per target, one matrix
                                per statistic. Each target-specific vector has two elements, and each element has size (ndo_φ, N_t), where N_t is the number of agents that prefer that target. There are two elements of the vector,
                                that contain the α and β parameters of the Gamma posterior over precision
    `gm_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative model
    `gp_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative process
    """

    # first get the function of a single target's belief and observation array, and fix the gm_params and gp_params arguments
    belief_update_f = (μ_t, φ_t, Γ_z_t) -> belief_updating_one_target_withGammaPlearning(μ_t, φ_t, Γ_z_t, gm_params, gp_params, D_shift_tdist)

    # then map the single-array version of the belief updating function across different targets
    mapped_results = map(belief_update_f, μ_t_target, φ_t_target, Γ_z_target)

    μ_t_all_targets = map(x -> x[1], mapped_results)
    ε_z_all_targets = map(x -> x[2], mapped_results)
    Γ_z_all_targets = map(x -> x[3], mapped_results)

    return μ_t_all_targets, ε_z_all_targets, Γ_z_all_targets

end

function run_belief_updating(μ::Array{Float32,1}, φ::Array{Float32,1}, dt::Float32, num_iter::Int64, κ_μ::Float32, ns_x::Int64, ndo_x::Int64,
                            ns_φ::Int64, ndo_φ::Int64, g::Function, ∂g∂x::Function, f::Function, ∂f∂x::Function, 𝚷_z::Array{Float32,2},
                            𝚷_ω::Array{Float32,2}, D_shift::Array{Float32,2})
    """
    This function runs gradient descent in generalised coordinates for a given timepoint to optimise the beliefs about
    hidden states μ and their higher orders of motion μ̃ in a moving frame of reference.
    ARGUMENTS
    =========
    `μ` - [Array{Float32,1}] - vector of beliefs about hidden states and their higher orders of motion, size (ns_x * ndo_x),
                                with hidden state dimensions being stored first/together, with chunks of subsequent hidden states
                                representing those same states at higher and higher generalised orders (e.g. velocity, acceleration, etc.)
    `φ` - [Array{Float32,1}] - vector of observations of hidden states and its observation higher orders of motion
    `dt` - [Int64] - step size for integration
    `num_iter` - [Int64] - number of variational iterations
    `κ_μ` - [Float32] - scalar learning rate, step size of gradient descent for beliefs μ̃
    `ns_x` - [Int64] - number of hidden state dimensions (at the 0-th order)
    `ndo_x`- [Int64] - number of generalised coordinates that beliefs are represented in
    `ns_φ` - [Int64] - number of observations dimensions (at the 0-th order)
    `ndo_φ`- [Int64] - number of generalised coordinates that observations are represented in
    `g` - [Function] - sensory likelihood function of the generative model, that maps beliefs about hidden states μ̃ to their expected sensory consequences g(μ̃)
    `∂g∂x` - [Function] - partial gradients of g with respect to its inputs x
    `f` - [Function] - process likelihood ('flow') function of the generative model, that maps beliefs about hidden states μ̃ to their expected motions Dμ̃
    `fg∂x` - [Function] - partial gradients of f with respect to its inputs x
    `𝚷_z` - [Array{Float32,2}] - sensory precisions (generalised, in space and time)
    `𝚷_ω` - [Array{Float32,2}] - process precisions (generalised, in space and time)
    `D_shift` - [Array{Float32,2}] - temporal shift matrix - shifts a vector of generalised coordinates 'up' by one order of motion
    """

    ε_z::Array{Float32,1} = zeros(Float32,ns_x*ndo_x) # running cache for storing instantaneous precision-weighted sensory prediction errors
    f_μ::Array{Float32,1} = zeros(Float32, ns_x*ndo_x) # running cache for storing instantaneous process expectations
    f_gradients::Array{Float32,1} = zeros(Float32, ns_x*ndo_x) # running cache for storing instantaneous process derivatives
    adjusted_dt::Float32 = Float32(dt/num_iter)

    for ii = 1:num_iter

        # sensory and process partial derivatives
        ∂g∂x_μ1 = ∂g∂x(μ[1:ns_x]);
        ∂g∂x_μ2 = ∂g∂x(μ[1:ns_x]);

        @views g_gradients = [∂g∂x_μ1; ∂g∂x_μ2];

        ∂f∂x_μ1 = ∂f∂x(μ[1:ns_x]);
        ∂f∂x_μ2 = ∂f∂x(μ[1:ns_x]);

        f_gradients[1:ns_x] = ∂f∂x_μ1 # gradients at order 0
        f_gradients[ns_x+1:(ns_x*(ndo_x-1))] = ∂f∂x_μ2 # gradients at order 1 (velocity)

        g_μ = [ g(μ[1:ns_φ]); ∂g∂x_μ1 .* μ[(ns_φ+1):(ns_x*ndo_φ)] ];
        s_pe = φ .- g_μ; # sensory prediction error: observations (φ) minus expectations (g_μ)

        # s_pe[vec(empty_sector_flags[:,:,n])] .= 0f0; % nullify the effect of sensory prediction errors from sectors that had no neighbors
        p_weighted_spe = 𝚷_z * s_pe;

        ε_z[1:(ns_φ*ndo_φ)] = g_gradients .* p_weighted_spe;

        f_μ[1:(ns_x*(ndo_x-1))] = [ f(μ[1:ns_x]); ∂f∂x_μ1 .* μ[(ns_x+1):(ns_x*(ndo_x-1))] ] # process expectations (the last order is all zeros - no 'same-level' information at highest order)

        p_pe = D_shift*μ .- f_μ; # process prediction errors (between the parallel/same level and the level above)

        # p_pe[vec(empty_sector_flags[:,:,n])] .= 0f0; % nullify the effect of process prediction errors from sectors that had no neighbors
        p_weighted_ppe = 𝚷_ω * p_pe;

        ε_ω = f_gradients .* p_weighted_ppe .- D_shift'*p_weighted_ppe;

        ∂μ∂t = D_shift*μ .+ ε_z .+ ε_ω;

        μ .+= (adjusted_dt .* κ_μ .* ∂μ∂t);

    end

    return μ, ε_z

end

function calculate_sector_hidden_states_new(ns_x, ndo_x, D, sector_idx, focal_dists, r, v, my_r, my_v, x_past)
    """
    Computes multivariate (`ns_x`) hidden states at `ndo_x` generalised orders
    ARGUMENTS
    =========
    `ns_x` - [Int64] - number of hidden state dimensions (at the 0-th order)
    `ndo_x`- [Int64] - number of generalised coordinates that hidden states are comprised of
    `D`    - [Int64] - spatial dimensionality
    `sector_idx` - [Vector{BitArray{1}}] - a vector of `ns_x` Boolean arrays (BitArrays) that signal the presence/absence of each other individual in the school in the i-th sector
    `focal_dists` - [Array{Float32,1}] - a vector of the distances between every other neighbour and the focal individual
    `r` - [Array{Float32,2}] - matrix of position vectors of all agents, size (`D`, `N`)
    `v` - [Array{Float32,2}] - matrix of velocity vectors of all agents, size (`D`, `N`)
    `my_r`- [Array{Float32,1}] - focal individual's position vector (length `D`)
    `my_v` - [Array{Float32,1}] - focal individual's velocity vector (length `D`)
    `x_past` - [Array{Float32,2}] - previous timestep's array of hidden states of size (`ns_x`, `ndo_x` - 1)
    """


    x_t = zeros(Float32, ns_x, ndo_x-1); # running cache for hidden states (used to generate observations)

    dh_dr_self_array = zeros(Float32, D, ns_x); # running cache for partial derivatives of hidden states with respect to position of focal particle
    empty_sector_flags = falses(ns_x, ndo_x); # running cache for flags that indicate absence of observations for each modality & dyamical order & individual

    # calculate hidden states
    for ns_i = 1:ns_x # loop through visual sectors

        if any(sector_idx[ns_i]) # if any individuals are within sector ns_i

            x_t[ns_i,1] = mean(focal_dists[sector_idx[ns_i]]); # calculates the mean distance between focal agent and the other agents within a given visual sector

            @views sector_r = r[:,sector_idx[ns_i]] .- my_r; # positions of neighbors within visual sector `ns_i`, centered on `my_r`
            @views sector_v = v[:,sector_idx[ns_i]]; # velocities of neighbors within visual_sector `ns_i`

            @views sector_r_norm = sector_r ./ sqrt.(sum(sector_r.^2,dims=1))
            dh_dr_self = -(mean(sector_r_norm,dims=2)); # partial derivative of distance with respect to my position vector
            dh_dr_others = sector_r_norm ./ sum(sector_idx[ns_i]);

            x_t[ns_i,2] = sum(dh_dr_self .* my_v) + sum(dh_dr_others .* sector_v)  # stores the velocity of the distance for visual_sector `ns_i`

            dh_dr_self_array[:,ns_i] = dh_dr_self;

        else  # if sector ns_i is empty
            empty_sector_flags[ns_i,:] .= true;
            try
                x_t[ns_i,:] = copy(x_past[ns_i,:]);
            catch
                x_t[ns_i,:] .= 0f0;
            end
            dh_dr_self_array[:,ns_i] = zeros(Float32,D);
        end # end of check about presence/absence of individuals within visual sector

    end # end of loop over visual sectors

    return x_t, dh_dr_self_array, empty_sector_flags

end


function calculate_sector_hidden_states(ns_x, ndo_x, D, sector_idx, focal_dists, neighbor_idx, other_r, other_v, my_r, my_v, x_past)
    """
    Computes multivariate (`ns_x`) hidden states at `ndo_x` generalised orders
    ARGUMENTS
    =========
    `ns_x` - [Int64] - number of hidden state dimensions (at the 0-th order)
    `ndo_x`- [Int64] - number of generalised coordinates that hidden states are comprised of
    `D`    - [Int64] - spatial dimensionality
    `sector_idx` - [Vector{BitArray{1}}] - a vector of `ns_x` Boolean arrays (BitArrays) that signal the presence/absence of each other individual in the school in the i-th sector
    `g` - [Function] - sensory likelihood function of the generative model, that maps beliefs about hidden states μ̃ to their expected sensory consequences g(μ̃)
    `∂g∂x` - [Function] - partial gradients of g with respect to its inputs x
    `f` - [Function] - process likelihood ('flow') function of the generative model, that maps beliefs about hidden states μ̃ to their expected motions Dμ̃
    `fg∂x` - [Function] - partial gradients of f with respect to its inputs x
    `𝚷_z` - [Array{Float32,2}] - sensory precisions (generalised, in space and time)
    `𝚷_ω` - [Array{Float32,2}] - process precisions (generalised, in space and time)
    `D_shift` - [Array{Float32,2}] - temporal shift matrix - shifts a vector of generalised coordinates 'up' by one order of motion
    """


    x_t = zeros(Float32, ns_x, ndo_x-1); # running cache for hidden states (used to generate observations)

    dh_dr_self_array = zeros(Float32, D, ns_x); # running cache for partial derivatives of hidden states with respect to position of focal particle
    empty_sector_flags = falses(ns_x, ndo_x); # running cache for flags that indicate absence of observations for each modality & dyamical order & individual

    # calculate hidden states
    for ns_i = 1:ns_x # loop through visual sectors

        if any(sector_idx[ns_i]) # if any individuals are within sector ns_i

            x_t[ns_i,1] = mean(focal_dists[neighbor_idx[sector_idx[ns_i]]]); # calculates the mean distance between focal agent and the other agents within a given visual sector

            @views sector_r = other_r[:,sector_idx[ns_i]] .- my_r; # positions of neighbors within visual sector `ns_i`, centered on `my_r`
            @views sector_v = other_v[:,sector_idx[ns_i]]; # velocities of neighbors within visual_sector `ns_i`

            @views sector_r_norm = sector_r ./ sqrt.(sum(sector_r.^2,dims=1))
            dh_dr_self = -(mean(sector_r_norm,dims=2)); # partial derivative of distance with respect to my position vector
            dh_dr_others = sector_r_norm ./ sum(sector_idx[ns_i]);

            x_t[ns_i,2] = sum(dh_dr_self .* my_v) + sum(dh_dr_others .* sector_v)  # stores the velocity of the distance for visual_sector `ns_i`

            dh_dr_self_array[:,ns_i] = dh_dr_self;

        else  # if sector ns_i is empty
            empty_sector_flags[ns_i,:] .= true;
            try
                x_t[ns_i,:] = copy(x_past[ns_i,:]);
            catch
                x_t[ns_i,:] .= 0f0;
            end
            dh_dr_self_array[:,ns_i] = zeros(Float32,D);
        end # end of check about presence/absence of individuals within visual sector

    end # end of loop over visual sectors

    return x_t, dh_dr_self_array, empty_sector_flags

end

function get_observations(ns_φ, ndo_φ, g, ∂g∂x, x_t, z_gp, noise_samples_φ)
    """
    Computes sensory observations at `ndo_φ` generalised orders
    """

    φ_t = zeros(Float32, ns_φ, ndo_φ); # observations for this agent (across sectors and dynamical orders)

    φ_t[:,1] = g(x_t[:,1]) + (z_gp[:,1] .* noise_samples_φ[:,1])
    φ_t[:,2] = (∂g∂x(x_t[:,1]) .* x_t[:,2]) + (z_gp[:,2] .* noise_samples_φ[:,2])

    return φ_t

end

function get_observations_vectorized(all_sector_h, all_sector_h_prime, noise_samples_φ, gp_params)
    """
    Computes sensory observations at 2 generalised orders (φ and φ_prime) using vectorized operations
    """

    φ = map(gp_params[:sampling_func_φ], all_sector_h, eachslice(noise_samples_φ,dims=3))
    h_x_hprime = map(gp_params[:h_by_hprime], all_sector_h, all_sector_h_prime)
    φprime = map(gp_params[:sampling_func_φprime], h_x_hprime, eachslice(noise_samples_φ,dims=3))

    return hcat(map((x,y) -> vcat(x, y), φ,  φprime)...)

end

function coincidence_experiments_2neighbours(spatial_smooth, temporal_smooth, rotation_angles, time_difference, initial_distance)
    """
    Function for running 'coincidence detection' experiments, wherein a focal active inference particle with position `my_pos` and `my_v`
    is lagging behind two 'virtual fish' that have a constant fixed velocity and distance from the focal individual.
    The focal agent's generative model has two sensory sectors, and is initialised using the following parameters:
    1. the spatial length scales of its sensory/process covariance matrices (the two entries of `spatial_smooth`);
    2. the temporal smoothness of its sensory/process autocovariance matrices (the two entries of `temporal_smooth`);

    Other parameters to specify are:
    a) the angles by which to rotate the two virtual fish during the stimulation period (stored in the respectively 1st and 2nd entries of `rotation_angles`)
    b) the difference between the time of rotation between the 1st and 2nd virtual fish, relative to the 1st VF (the 'leftmost' virtual fish). A positive number means
    the 1st virtual fish turns first
    c) the initial distance between the focal individual and the two virtual fish

    """
    my_pos = [0f0, 0f0]
    my_v = [0f0, 1f0]

    forward_shift  = 0.5f0
    horizontal_shift = sqrt(initial_distance^2 -  forward_shift^2)

    neighbour1_pos = [my_pos[1]-horizontal_shift, my_pos[2] + forward_shift] # to the left of focal individual
    neighbour1_v = [0f0, 1f0]

    neighbour2_pos = [my_pos[1]+horizontal_shift, my_pos[2] + forward_shift] # to the right of focal individual
    neighbour2_v = [0f0, 1f0]

    ns_φ, ndo_φ, ns_x, ndo_x = 2, 2, 2, 3

    sector_angles = [120f0, 0f0, 360f0 - 120f0]

    ls_z, ls_ω = spatial_smooth[1], spatial_smooth[2]
    s_z, s_ω = temporal_smooth[1], temporal_smooth[2]

    temporal_correlations_z = GMUtils.compute_temporal_precisions(ndo_φ, s_z)[1]
    temporal_correlations_ω = GMUtils.compute_temporal_precisions(ndo_x, s_ω)[1]

    spatial_z = GMUtils.compute_spatial_precisions(ns_φ, ls_z, 1f0)[1]
    spatial_ω = GMUtils.compute_spatial_precisions(ns_φ, ls_ω, 1f0)[1]

    𝚷_z = kron(temporal_correlations_z, spatial_z)
    𝚷_ω = kron(temporal_correlations_ω, spatial_ω)

    D_shift = diagm(ns_x => ones(Float32,ndo_x*ns_x - ns_x));

    D = 2

    T = 50
    dt = 0.01

    t_axis = 0:dt:T; # time axis in seconds
    T_sim = length(t_axis); # length of time-axis in number of samples

    stim_time = round(Int64, T_sim/2)
    stim_duration = 1000

    stim_window_1 = stim_time:(stim_time + stim_duration)
    stim_window_2 = stim_window_1 .+ time_difference

    EM_scalar = sqrt(dt)
    z_gp = EM_scalar .* 0.1f0 * ones(Float32,ns_φ,ndo_φ)
    z_action = EM_scalar .* (0.01f0.* ones(Float32,D))

    pos_hist = zeros(Float32, D, T_sim)
    v_hist = zeros(Float32, D, T_sim)

    pos_hist[:,1] = my_pos
    v_hist[:,1] = my_v

    x_hist     = zeros(Float32, ns_x, ndo_x-1, T_sim) # history of hidden states
    φ_hist     = zeros(Float32, ns_φ, ndo_φ, T_sim) # history of observations
    μ_hist     = zeros(Float32,(ns_x*ndo_x), T_sim) # history of beliefs
    dF_dv_hist = zeros(Float32, D, ns_φ, T_sim) # history of partial derivatives of F w.r.t to action, divided up by sector

    x_t = zeros(Float32, ns_x, ndo_x-1); # running cache for hidden states (used to generate observations)
    φ_t = zeros(Float32, ns_φ, ndo_φ); # running cache for observations

    μ_t = zeros(Float32,ns_x*ndo_x) # running cache for storing instantaneous beliefs for a given individual

    rotation_matrices = generate_rotation_matrices(sector_angles)

    other_pos_hist = zeros(Float32, D, 2, T_sim)
    other_pos_hist[:,1,1] = neighbour1_pos
    other_pos_hist[:,2,1] = neighbour2_pos

    other_v_hist = zeros(Float32, D, 2, T_sim)
    other_v_hist[:,1,1] = neighbour1_v
    other_v_hist[:,2,1] = neighbour2_v

    dh_dr_self_hist = zeros(Float32, D, ns_x, T_sim); # array storing history of partial derivatives of hidden states with respect to position of focal particle
    ∂F∂φprime_hist = zeros(Float32, ns_φ, T_sim)       # array storing history of prediction errors of velocity observation

    g(x) = x # sensory function
    ∂g∂x(x) = 1f0 # its derivative
    f(x) = -1f0 .* (x .- 1f0); # flow function
    ∂f∂x(x) = -1f0; # its derivative

    κ_μ = 0.1f0
    κ_a = 0.1f0

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        ####### UPDATE GENERATIVE PROCESS #######

        if t in stim_window_1
            neighbour1_v = [sind(rotation_angles[1]), cosd(rotation_angles[1]) ]
        elseif t > stim_window_1[end]
            neighbour1_v = [0f0, 1f0]
        end

        if t in stim_window_2
            neighbour2_v = [sind(rotation_angles[2]), cosd(rotation_angles[2])]
        elseif t > stim_window_1[end]
            neighbour2_v = [0f0, 1f0]
        end

        # update position using last timestep's velocity
        # my_pos = my_pos .+ (dt .* my_v) .+ (z_action .* randn(D));
        my_pos = my_pos .+ (dt .* my_v) # no action noise version
        neighbour1_pos = neighbour1_pos .+ (dt .* neighbour1_v);
        neighbour2_pos = neighbour2_pos .+ (dt .* neighbour2_v);

        other_r =  hcat([neighbour1_pos, neighbour2_pos]...)
        other_v =  hcat([neighbour1_v, neighbour2_v]...)

        sector_idx = GeoUtils.assign_visfield_ids(my_pos, my_v, other_r, ns_x, rotation_matrices, 10f0); # returns a vector of BitArrays, one for each visual sector, that index whether neighbor particles are in the corresponding visual sector

        distances = [norm(my_pos - other_r[:,j]) for j in 1:2]
        # calculate hidden states
        x_t, dh_dr_self, empty_sector_flags = calculate_sector_hidden_states(ns_x, ndo_x, D, sector_idx, distances, 1:2, other_r, other_v, my_pos, my_v, x_hist[:,:,t-1])

        # generate observations from hidden states
        φ_t = get_observations(ns_φ, ndo_φ, g, ∂g∂x, x_t, z_gp, randn(ns_φ,ndo_φ))

        ####### NOTE!!! about converting from matrix [ns_x, ndo_x] representation to vectorised [ns_x * ndo_x,] representation: ##############
        # we flatten arrays in the [ns_x, ndo_x] format into a single long vector, wherehidden state dimensions within the first dynamical order are listed first,
        # before the next dynamical order starts. For example: if a hidden state dimensionality of 5, and 2 dynamical orders: then entries
        # 1 - 5 of the vectorised representation will represent all the states for the first dynamical order, and entries 6 - 10 will store the states for the second dynamical order, etc.
        # We do it this way arbitrarily, but based on the fact that the precision matrices are stored this way (smaller block diagonals represent single dynamical orders, for all hidden states,
        # whereas moving from one block diagonal to the next corresponds to changing dynamical orders).
        ######################################################################################################################################

        vectorized_φ = vec(φ_t);

        if t == 2
            μ_t[1:(ns_φ*ndo_φ)] =  vectorized_φ # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
            μ_t[(ns_φ*ndo_φ +1):end] .= 0f0 # do this to make sure acceleration beliefs from agent n-1 aren't 'shared' to agent n
        else
            μ_t = copy(μ_hist[:,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
        end

        ####### Update beliefs using generalised predictive coding #######

        ε_z = zeros(Float32,ns_x*ndo_x) # running cache for storing instantaneous precision-weighted sensory prediction errors
        f_μ = zeros(Float32, ns_x*ndo_x) # running cache for storing instantaneous process expectations
        f_gradients= zeros(Float32, ns_x*ndo_x) # running cache for storing instantaneous process derivatives

        # sensory and process partial derivative

        f_gradients[1:ns_x] .= -1f0 # gradients at order 0
        f_gradients[ns_x+1:(ns_x*(ndo_x-1))] .= -1f0 # gradients at order 1 (velocity)

        g_μ = [ g(μ_t[1:ns_φ]); μ_t[(ns_φ+1):(ns_x*ndo_φ)] ];
        s_pe = vectorized_φ .- g_μ; # sensory prediction error: observations (φ) minus expectations (g_μ)

        # s_pe[vec(empty_sector_flags)] .= 0f0; % nullify the effect of sensory prediction errors from sectors that had no neighbors
        p_weighted_spe = 𝚷_z * s_pe;

        ε_z[1:(ns_φ*ndo_φ)] = p_weighted_spe;

        f_μ[1:(ns_x*(ndo_x-1))] = [ f(μ_t[1:ns_x]); -1.0f0 .* μ_t[(ns_x+1):(ns_x*(ndo_x-1))] ] # process expectations (the last order is all zeros - no 'same-level' information at highest order)

        p_pe = D_shift*μ_t .- f_μ; # process prediction errors (between the parallel/same level and the level above)

        # p_pe[vec(empty_sector_flags)] .= 0f0; % nullify the effect of process prediction errors from sectors that had no neighbors
        p_weighted_ppe = 𝚷_ω * p_pe;

        ε_ω = f_gradients .* p_weighted_ppe .- D_shift'*p_weighted_ppe;

        ∂μ∂t = D_shift*μ_t .+ ε_z .+ ε_ω;

        μ_t .+= (dt .* κ_μ .* ∂μ∂t);

        μ_hist[:,t] = copy(μ_t)

        ∂F∂φprime = ε_z[(ns_φ+1):(ns_φ*ndo_φ)]; # the second ns_φ elements of the vector of sensory prediction errors correspond to the different elements of ∂F∂φprime

        ∂φprime_∂v = dh_dr_self;
        ∂F∂v = ∂φprime_∂v*∂F∂φprime; # if you're not keeping track of the individual partial derivatives, then you can do it all in one dot product, like this

        my_v .-= (κ_a .* ∂F∂v); # update velocity

        my_v = my_v./ sqrt.(sum(my_v.^2)); # normalize velocities
        x_hist[:,:,t] .= copy(x_t)
        φ_hist[:,:,t] .= copy(φ_t)

        pos_hist[:,t] = copy(my_pos)
        v_hist[:,t] = copy(my_v)

        other_pos_hist[:,:,t] = copy(other_r)
        other_v_hist[:,:,t] = copy(other_v)

        dh_dr_self_hist[:,:,t] = copy(dh_dr_self)
        ∂F∂φprime_hist[:,t] = copy(∂F∂φprime)

    end # end of loop over time

    return pos_hist, v_hist, other_pos_hist, other_v_hist, dh_dr_self_hist, ∂F∂φprime_hist

end

function coincidence_experiments_2neighbours_smoothturning(spatial_smooth, temporal_smooth, rotation_angles, turning_rates, time_difference, initial_distance)
    """
    Function for running 'coincidence detection' experiments, wherein a focal active inference particle with position `my_pos` and `my_v`
    is lagging behind two 'virtual fish' that have a constant fixed velocity and distance from the focal individual.
    The focal agent's generative model has two sensory sectors, and is initialised using the following parameters:
    1. the spatial length scales of its sensory/process covariance matrices (the two entries of `spatial_smooth`);
    2. the temporal smoothness of its sensory/process autocovariance matrices (the two entries of `temporal_smooth`);

    Other parameters to specify are:
    a) the angles by which to rotate the two virtual fish during the stimulation period (stored in the respectively 1st and 2nd entries of `rotation_angles`)
    b) the difference between the time of rotation between the 1st and 2nd virtual fish, relative to the 1st VF (the 'leftmost' virtual fish). A positive number means
    the 1st virtual fish turns first
    c) the initial distance between the focal individual and the two virtual fish

    !!! NEW VERSION !!!
    Now the 'virtual' leader fish change gradually rather than instantly to their final turn angles
    """
    my_pos = [0f0, 0f0]
    my_v = [0f0, 1f0]

    forward_shift  = 0.5f0
    horizontal_shift = sqrt(initial_distance^2 -  forward_shift^2)

    neighbour1_pos = [my_pos[1]-horizontal_shift, my_pos[2] + forward_shift] # to the left of focal individual
    neighbour1_v = [0f0, 1f0]
    neighbour_1_angle_sequence = collect(0:turning_rates[1]:rotation_angles[1])

    neighbour2_pos = [my_pos[1]+horizontal_shift, my_pos[2] + forward_shift] # to the right of focal individual
    neighbour2_v = [0f0, 1f0]
    neighbour_2_angle_sequence = collect(0:turning_rates[2]:rotation_angles[2])

    angle_seq_idx_1, angle_seq_idx_2 = 1, 1

    ns_φ, ndo_φ, ns_x, ndo_x = 2, 2, 2, 3

    sector_angles = [120f0, 0f0, 360f0 - 120f0]

    ls_z, ls_ω = spatial_smooth[1], spatial_smooth[2]
    s_z, s_ω = temporal_smooth[1], temporal_smooth[2]

    temporal_correlations_z = GMUtils.compute_temporal_precisions(ndo_φ, s_z)[1]
    temporal_correlations_ω = GMUtils.compute_temporal_precisions(ndo_x, s_ω)[1]

    spatial_z = GMUtils.compute_spatial_precisions(ns_φ, ls_z, 1f0)[1]
    spatial_ω = GMUtils.compute_spatial_precisions(ns_φ, ls_ω, 1f0)[1]

    𝚷_z = kron(temporal_correlations_z, spatial_z)
    𝚷_ω = kron(temporal_correlations_ω, spatial_ω)

    D_shift = diagm(ns_x => ones(Float32,ndo_x*ns_x - ns_x));

    D = 2

    T = 20
    dt = 0.01

    t_axis = 0:dt:T; # time axis in seconds
    T_sim = length(t_axis); # length of time-axis in number of samples

    stim_time = round(Int64, T_sim/2)
    stim_duration = 500

    stim_window_1 = stim_time:(stim_time + stim_duration)
    stim_window_2 = stim_window_1 .+ time_difference

    EM_scalar = sqrt(dt)
    z_gp = EM_scalar .* 0.1f0 * ones(Float32,ns_φ,ndo_φ)
    z_action = EM_scalar .* (0.01f0.* ones(Float32,D))

    pos_hist = zeros(Float32, D, T_sim)
    v_hist = zeros(Float32, D, T_sim)

    pos_hist[:,1] = my_pos
    v_hist[:,1] = my_v

    x_hist     = zeros(Float32, ns_x, ndo_x-1, T_sim) # history of hidden states
    φ_hist     = zeros(Float32, ns_φ, ndo_φ, T_sim) # history of observations
    μ_hist     = zeros(Float32,(ns_x*ndo_x), T_sim) # history of beliefs
    dF_dv_hist = zeros(Float32, D, ns_φ, T_sim) # history of partial derivatives of F w.r.t to action, divided up by sector

    x_t = zeros(Float32, ns_x, ndo_x-1); # running cache for hidden states (used to generate observations)
    φ_t = zeros(Float32, ns_φ, ndo_φ); # running cache for observations

    μ_t = zeros(Float32,ns_x*ndo_x) # running cache for storing instantaneous beliefs for a given individual

    rotation_matrices = generate_rotation_matrices(sector_angles)

    other_pos_hist = zeros(Float32, D, 2, T_sim)
    other_pos_hist[:,1,1] = neighbour1_pos
    other_pos_hist[:,2,1] = neighbour2_pos

    other_v_hist = zeros(Float32, D, 2, T_sim)
    other_v_hist[:,1,1] = neighbour1_v
    other_v_hist[:,2,1] = neighbour2_v

    dh_dr_self_hist = zeros(Float32, D, ns_x, T_sim); # array storing history of partial derivatives of hidden states with respect to position of focal particle
    ∂F∂φprime_hist = zeros(Float32, ns_φ, T_sim)       # array storing history of prediction errors of velocity observation

    g(x) = x # sensory function
    ∂g∂x(x) = 1f0 # its derivative
    f(x) = -1f0 .* (x .- 1f0); # flow function
    ∂f∂x(x) = -1f0; # its derivative

    κ_μ = 0.1f0
    κ_a = 0.1f0


    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        ####### UPDATE GENERATIVE PROCESS #######

        if t in stim_window_1
            angle_1 = neighbour_1_angle_sequence[angle_seq_idx_1]
            neighbour1_v = [sind(angle_1), cosd(angle_1)]
            angle_seq_idx_1 += 1

            if angle_seq_idx_1 > length(neighbour_1_angle_sequence)
                angle_seq_idx_1 = length(neighbour_1_angle_sequence)
            end
        elseif t > stim_window_1[end]
            neighbour1_v = [0f0, 1f0]
        end

        if t in stim_window_2

            angle_2 = neighbour_2_angle_sequence[angle_seq_idx_2]
            neighbour2_v = [sind(angle_2), cosd(angle_2)]
            angle_seq_idx_2 += 1

            if angle_seq_idx_2 > length(neighbour_2_angle_sequence)
                angle_seq_idx_2 = length(neighbour_2_angle_sequence)
            end

        elseif t > stim_window_1[end]
            neighbour2_v = [0f0, 1f0]
        end

        # update position using last timestep's velocity
        # my_pos = my_pos .+ (dt .* my_v) .+ (z_action .* randn(D));
        my_pos = my_pos .+ (dt .* my_v) # no action noise version
        neighbour1_pos = neighbour1_pos .+ (dt .* neighbour1_v);
        neighbour2_pos = neighbour2_pos .+ (dt .* neighbour2_v);

        other_r =  hcat([neighbour1_pos, neighbour2_pos]...)
        other_v =  hcat([neighbour1_v, neighbour2_v]...)

        sector_idx = GeoUtils.assign_visfield_ids(my_pos, my_v, other_r, ns_x, rotation_matrices, 10f0); # returns a vector of BitArrays, one for each visual sector, that index whether neighbor particles are in the corresponding visual sector

        distances = [norm(my_pos - other_r[:,j]) for j in 1:2]
        # calculate hidden states
        x_t, dh_dr_self, empty_sector_flags = calculate_sector_hidden_states(ns_x, ndo_x, D, sector_idx, distances, 1:2, other_r, other_v, my_pos, my_v, x_hist[:,:,t-1])

        # generate observations from hidden states
        φ_t = get_observations(ns_φ, ndo_φ, g, ∂g∂x, x_t, z_gp, randn(ns_φ,ndo_φ))

        ####### NOTE!!! about converting from matrix [ns_x, ndo_x] representation to vectorised [ns_x * ndo_x,] representation: ##############
        # we flatten arrays in the [ns_x, ndo_x] format into a single long vector, wherehidden state dimensions within the first dynamical order are listed first,
        # before the next dynamical order starts. For example: if a hidden state dimensionality of 5, and 2 dynamical orders: then entries
        # 1 - 5 of the vectorised representation will represent all the states for the first dynamical order, and entries 6 - 10 will store the states for the second dynamical order, etc.
        # We do it this way arbitrarily, but based on the fact that the precision matrices are stored this way (smaller block diagonals represent single dynamical orders, for all hidden states,
        # whereas moving from one block diagonal to the next corresponds to changing dynamical orders).
        ######################################################################################################################################

        vectorized_φ = vec(φ_t);

        if t == 2
            μ_t[1:(ns_φ*ndo_φ)] =  vectorized_φ # if we're at the first (really, second) timestep of the simulation, set beliefs equal to the observations
            μ_t[(ns_φ*ndo_φ +1):end] .= 0f0 # do this to make sure acceleration beliefs from agent n-1 aren't 'shared' to agent n
        else
            μ_t = copy(μ_hist[:,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
        end

        ####### Update beliefs using generalised predictive coding #######

        ε_z = zeros(Float32,ns_x*ndo_x) # running cache for storing instantaneous precision-weighted sensory prediction errors
        f_μ = zeros(Float32, ns_x*ndo_x) # running cache for storing instantaneous process expectations
        f_gradients= zeros(Float32, ns_x*ndo_x) # running cache for storing instantaneous process derivatives

        # sensory and process partial derivative

        f_gradients[1:ns_x] .= -1f0 # gradients at order 0
        f_gradients[ns_x+1:(ns_x*(ndo_x-1))] .= -1f0 # gradients at order 1 (velocity)

        g_μ = [ g(μ_t[1:ns_φ]); μ_t[(ns_φ+1):(ns_x*ndo_φ)] ];
        s_pe = vectorized_φ .- g_μ; # sensory prediction error: observations (φ) minus expectations (g_μ)

        # s_pe[vec(empty_sector_flags)] .= 0f0; % nullify the effect of sensory prediction errors from sectors that had no neighbors
        p_weighted_spe = 𝚷_z * s_pe;

        ε_z[1:(ns_φ*ndo_φ)] = p_weighted_spe;

        f_μ[1:(ns_x*(ndo_x-1))] = [ f(μ_t[1:ns_x]); -1.0f0 .* μ_t[(ns_x+1):(ns_x*(ndo_x-1))] ] # process expectations (the last order is all zeros - no 'same-level' information at highest order)

        p_pe = D_shift*μ_t .- f_μ; # process prediction errors (between the parallel/same level and the level above)

        # p_pe[vec(empty_sector_flags)] .= 0f0; % nullify the effect of process prediction errors from sectors that had no neighbors
        p_weighted_ppe = 𝚷_ω * p_pe;

        ε_ω = f_gradients .* p_weighted_ppe .- D_shift'*p_weighted_ppe;

        ∂μ∂t = D_shift*μ_t .+ ε_z .+ ε_ω;

        μ_t .+= (dt .* κ_μ .* ∂μ∂t);

        μ_hist[:,t] = copy(μ_t)

        ∂F∂φprime = ε_z[(ns_φ+1):(ns_φ*ndo_φ)]; # the second ns_φ elements of the vector of sensory prediction errors correspond to the different elements of ∂F∂φprime

        ∂φprime_∂v = dh_dr_self;
        ∂F∂v = ∂φprime_∂v*∂F∂φprime; # if you're not keeping track of the individual partial derivatives, then you can do it all in one dot product, like this

        my_v .-= (κ_a .* ∂F∂v); # update velocity

        my_v = my_v./ sqrt.(sum(my_v.^2)); # normalize velocities
        x_hist[:,:,t] .= copy(x_t)
        φ_hist[:,:,t] .= copy(φ_t)

        pos_hist[:,t] = copy(my_pos)
        v_hist[:,t] = copy(my_v)

        other_pos_hist[:,:,t] = copy(other_r)
        other_v_hist[:,:,t] = copy(other_v)

        dh_dr_self_hist[:,:,t] = copy(dh_dr_self)
        ∂F∂φprime_hist[:,t] = copy(∂F∂φprime)

    end # end of loop over time

    return pos_hist, v_hist, other_pos_hist, other_v_hist, dh_dr_self_hist, ∂F∂φprime_hist, μ_hist

end

end


## UNFINISHED FUNCTIONS


function run_belief_updating_tdist_withPlearning(μ_t_target, φ_t_target, 𝚷_z_target, gm_params, gp_params, D_shift_tdist)
    """
    This function runs gradient descent in generalised coordinates for a given timepoint to optimise the beliefs about
    hidden states μ and their higher orders of motion μ̃ in a moving frame of reference, for difference targets (independent elements of μ_t_target).
    We also update the precisions using exact Bayesian updating scheme for gamma conjugate priors described in Baioumy et al. (2022):  "Precision from History: Fault-tolerant Control
    for Sensory Faults via Bayesian Inference and Geometric Weighting"
    ARGUMENTS
    =========
    `μ` - [Vector{Array{Float32,2}}] - vector containing matrices of beliefs about tdist hidden states and their higher orders of motion across individuals, one matrix per target. Each sub-matrix of the vector is
                                of size (ndo_x, N) with rows indexing higher generalised orders (e.g. velocity, acceleration, etc.)
    `φ` - [Vector{Array{Float32,2}}] - vector containing matrices of observations of tdist hidden states and their higher orders of motion across individuals, one matrix per target.
                                Each sub-matrix of the vector is of size (ndo_φ, N) with rows indexing higher generalised orders (e.g. velocity, acceleration, etc.)
    `𝚷_z_target` - [Vector{Array{Float32,2}}] - vector whose elements contain collections of beliefs about tdist sensory precisions and their higher orders of motion, across individuals. One tensor per target, one matrix
                                per individual. Each target-specific tensor is of size (ndo_φ, ndo_φ, N_t), where N_t is the number of agents that prefer that target
    `gm_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative model
    `gp_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative process
    """

    # first get the function of a single target's belief and observation array, and fix the gm_params and gp_params arguments
    belief_update_f = (μ_t, φ_t, 𝚷_z_t) -> belief_updating_one_target_withPlearning(μ_t, φ_t, 𝚷_z_t, gm_params, gp_params, D_shift_tdist)

    # then map the single-array version of the belief updating function across different targets
    mapped_results = map(belief_update_f, μ_t_target, φ_t_target, 𝚷_z_target)

    μ_t_all_targets = map(x -> x[1], mapped_results)
    ε_z_all_targets = map(x -> x[2], mapped_results)
    𝚷_z_all_targets = map(x -> x[3], mapped_results)

    return μ_t_all_targets, ε_z_all_targets, 𝚷_z_all_targets

end

function belief_updating_one_target_withPlearning(μ, φ, 𝚷_z, gm_params, gp_params, D_shift)

    """
    This function runs gradient descent in generalised coordinates for a given timepoint to optimise the beliefs about
    hidden states μ that represent the distance to a target and their higher orders of motion μ̃ in a moving frame of reference.
    ARGUMENTS
    =========
    `μ` - [Array{Float32,2}] - matrix of beliefs about hidden states (about stargets) and their higher orders of motion for each individual, of size (ndo_x, N),
                                with rows of subsequent hidden states indexing higher and higher generalised orders (e.g. velocity, acceleration, etc.)
    `φ` - [Array{Float32,2}] - matrix of observations of hidden states and its observation at higher orders of motion for each individual, of size (ndo_φ, N)
     𝚷_z - [Array{Float32,3}] - tensor of observation precision matrices across higher orders of motion for each individual, of size (ndo_φ, ndo_φ, N)
    `gm_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative model
    `gp_params` [Dict{Symbol, Any}] - dictionary containing parameters & functions relevant to the generative process
    `D_shift`   [Matrix{Float32, 2}] - shift operator that shifts 'up' a vector of generalised coordinates, so that now (D_shift*mu)[i] = mu[i+1], and mu[end] = 0.0
    """

    N = size(μ,2)
    dt = gp_params[:dt]

    ns_x = gm_params[:ns_x]
    ndo_x = gm_params[:ndo_x]

    num_iter = gm_params[:num_iter]
    κ_μ = gm_params[:κ_μ]

    f = gm_params[:flow_func_tdist]
    ∂f∂x = gm_params[:flow_func_tdistprime]

    # 𝚷_z = gm_params[:𝚷_z_tdist] # comment this out, now it's an input
    𝚷_ω = gm_params[:𝚷_ω_tdist]

    f_gradients::Array{Float32,2} = zeros(Float32, ndo_x, N) # running cache for storing instantaneous process derivatives

    ε_z::Array{Float32,2} = zeros(Float32, ndo_x, N) # running cache for storing instantaneous precision-weighted sensory prediction errors

    f_μ::Array{Float32,2} = zeros(Float32, ndo_x, N) # running cache for storing instantaneous process expectations

    for ii = 1:num_iter

        # the partial derivatives of the sensory mapping (g) are all gonna be 1.0 in this case
        # since we don't have a nonlinear observation function

        # ∂g∂x_μ1 = ∂g∂x(μ[1:ns_x,:]);
        # ∂g∂x_μ2 = ∂g∂x(μ[1:ns_x,:]);
        #
        # @views g_gradients = [∂g∂x_μ1; ∂g∂x_μ2];

        ∂f∂x_μ1 = ∂f∂x.(μ[1,:]);
        ∂f∂x_μ2 = ∂f∂x.(μ[1,:]);

        f_gradients[1,:] = ∂f∂x_μ1 # gradients at order 0
        f_gradients[2,:] = ∂f∂x_μ2 # gradients at order 1 (velocity)

        # g_μ = [ g(μ[1:ns_φ,:]); ∂g∂x_μ1 .* μ[(ns_φ+1):(ns_x*ndo_φ),:] ];
        # can simplify this since we know g(x) is identity
        # g_μ = [ μ[1,:]';  μ[2,:]'];
        g_μ = copy(μ[1:2,:])

        s_pe = φ .- g_μ; # sensory prediction error: observations (φ) minus expectations (g_μ)

        # if 𝚷_z is d x d x N, and s_pe is d x 1 x N, then we can use NNLib.batched_mul to multiply each precision matrix with each individual's sensory precision error
        p_weighted_spe = dropdims(batched_mul(𝚷_z, reshape(s_pe, ndo_x-1, 1, N)), dims = 2)

        # ε_z[1:2,:] = g_gradients .* p_weighted_spe;
        ε_z[1:2,:] = p_weighted_spe;

        f_μ[1:2,:] = [ f(μ[1,:])'; (∂f∂x_μ1 .* μ[2,:])' ] # process expectations (the last order is all zeros - no 'same-level' information at highest order)

        p_pe = D_shift*μ .- f_μ; # process prediction errors (between the parallel/same level and the level above)

        p_weighted_ppe = 𝚷_ω * p_pe;

        ε_ω = f_gradients .* p_weighted_ppe .- D_shift'*p_weighted_ppe;

        ∂μ∂t = D_shift*μ .+ ε_z .+ ε_ω;

        μ .+= κ_μ .* ∂μ∂t;

    end

    # LEFT OFF HERE put update here
    # UNFINISHED

    return μ, ε_z, 𝚷_z

end
