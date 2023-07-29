module SimUtils

include("GeoUtils.jl")
include("NoiseUtils.jl")
include("AnalysisUtils.jl")
include("SimUtils.jl")

using LinearAlgebra
using Statistics
using Distances


function run_simulation_externalnoise(r::Array{Float32,3}, v::Array{Float32,3}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any},
    noise_cache::Array{Float32,1}; save_dFdv::Bool=false)
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

    # create the shift matrix
    D_shift::Array{Float32,2} = diagm(ns_x => ones(Float32,ndo_x*ns_x - ns_x));

    # create rotation matrices
    rotation_matrices = SimUtils.generate_rotation_matrices(gp_params[:sector_angles])

    # create distance matrix for storing instantaneous distances between agents
    dist_matrix::Array{Float32,2} = zeros(Float32,N,N)

    x_hist = zeros(Float32, ns_x, ndo_x-1, N, T_sim) # history of hidden states
    φ_hist = zeros(Float32, ns_φ, ndo_φ, N, T_sim) # history of observations
    μ_x    = zeros(Float32,(ns_x*ndo_x), N, T_sim) # history of beliefs
    dF_dv_hist = zeros(Float32,D, ns_φ, N, T_sim) # history of partial derivatives of F w.r.t to action, divided up by sector

    x_t = zeros(Float32, ns_x, ndo_x-1, N); # running cache for hidden states (used to generate observations)
    φ_t = zeros(Float32, ns_φ, ndo_φ, N); # running cache for observations

    dh_dr_self_array = zeros(Float32, D, ns_x, N); # running cache for partial derivatives of hidden states with respect to position of focal particle
    empty_sector_flags = falses(ns_x, ndo_x, N); # running cache for flags that indicate absence of observations for each modality & dyamical order & individual

    μ_x_n_t = zeros(Float32,ns_x*ndo_x) # running cache for storing instantaneous beliefs for a given individual

    last_idx = 1; # index for updating the noise caché

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep


        noise_samples_action = noise_cache[last_idx:(last_idx+(D*N)-1)]
        noise_samples_action = reshape(noise_samples_action,(D,N))
        last_idx += (D*N)

        ####### UPDATE GENERATIVE PROCESS #######

        # update position using last timestep's velocity
        r[:,:,t] = r[:,:,t-1] .+ (dt .* v[:,:,t-1]) + (z_action .* noise_samples_action);
        r_t = @view r[:,:,t];

        v_past = @view v[:,:,t-1];
        v[:,:,t] = copy(v[:,:,t-1]); # carry forward last timestep's velocity to the current timestep

        # calculate distance between all particles
        # [dist_matrix[i,j] = norm(r_t[:,i] .- r_t[:,j]) for i = 1:N, j = 1:N]
        pairwise!(dist_matrix, Euclidean(), r_t, dims=2); # this goes way faster than the above approach

        # noise_samples_φ = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))
        noise_samples_φ = noise_cache[last_idx:(last_idx+(N*ns_φ*ndo_φ)-1)]
        noise_samples_φ = reshape(noise_samples_φ, (ns_φ, ndo_φ, N))
        last_idx += (N*ns_φ*ndo_φ);

        for n = 1:N # loop over individuals

            ####### get hidden states and observations #######

            neighbor_idx = findall(1:N .!== n) # all the particles besides me

            @views my_r, my_v = r_t[:,n], v_past[:,n] # focal particle's position and velocity

            @views other_r, other_v  = r_t[:,neighbor_idx], v_past[:,neighbor_idx] # all other particles' positions and velocities

            sector_idx = GeoUtils.assign_visfield_ids(my_r, my_v, other_r, ns_x, rotation_matrices, dist_thr); # returns a vector of BitArrays, one for each visual sector, that index whether neighbor particles are in the corresponding visual sector

            # calculate hidden states
            x_t[:,:,n], dh_dr_self_array[:,:,n], empty_sector_flags[:,:,n] =  SimUtils.calculate_sector_hidden_states(ns_x, ndo_x, D, sector_idx, dist_matrix[n,:], neighbor_idx, other_r, other_v, my_r, my_v, x_hist[:,:,n,t-1])

            # generate observations from hidden states
            @views hidden_states_n, noise_n = x_t[:,:,n], noise_samples_φ[:,:,n]
            φ_t[:,:,n] = SimUtils.get_observations(ns_φ, ndo_φ, g, ∂g∂x, hidden_states_n, z_gp, noise_n)

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
                μ_x_n_t[(ns_φ*ndo_φ +1):end] .= 0f0 # do this to make sure acceleration beliefs from agent n-1  aren't 'shared' to agent n
            else
                μ_x_n_t = copy(μ_x[:,n,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
            end

            μ_x[:,n,t], ε_z = SimUtils.run_belief_updating(μ_x_n_t, vectorized_φ, dt, num_iter, κ_μ,
                                                 ns_x, ndo_x, ns_φ, ndo_φ,
                                                 g, ∂g∂x, f, ∂f∂x, 𝚷_z, 𝚷_ω, D_shift)

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

        end # end of loop over individuals

        v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities
        x_hist[:,:,:,t] .= copy(x_t)
        φ_hist[:,:,:,t] .= copy(φ_t)

    end # end of loop over time

    results_dict = Dict(:r => r, :v => v, :x_hist => x_hist, :φ_hist => φ_hist, :μ_x => μ_x, :dF_dv_hist => dF_dv_hist)

    return results_dict
end # end of run_simulation function

function run_simulation_with_perturbations(r::Array{Float32,3}, v::Array{Float32,3}, gm_params::Dict{Symbol,Any}, gp_params::Dict{Symbol,Any},
    perturbation_times; num_to_perturb::Int64 = 5, save_dFdv::Bool=false)
    """
    This function runs a single realization of multimodal schooling where external perturbations are added to certain members in the front of the school.
    """

    # unpack global constant parameters from `gp_params`
    T_sim = gp_params[:T_sim]
    N = gp_params[:N]
    D = gp_params[:D]
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

    # create the shift matrix
    D_shift::Array{Float32,2} = diagm(ns_x => ones(Float32,ndo_x*ns_x - ns_x));

    # create rotation matrices
    rotation_matrices = SimUtils.generate_rotation_matrices(gp_params[:sector_angles])

    # create noise structure
    noise_struct = NoiseUtils.NoiseStruct(num_samples = (T_sim*N*D*ns_φ*ndo_φ*N) )

    # create distance matrix for storing instantaneous distances between agents
    dist_matrix::Array{Float32,2} = zeros(Float32,N,N)

    x_hist = zeros(Float32, ns_x, ndo_x-1, N, T_sim) # history of hidden states
    φ_hist = zeros(Float32, ns_φ, ndo_φ, N, T_sim) # history of observations
    μ_x    = zeros(Float32,(ns_x*ndo_x), N, T_sim) # history of beliefs
    dF_dv_hist = zeros(Float32,D, ns_φ, N, T_sim) # history of partial derivatives of F w.r.t to action, divided up by sector

    x_t = zeros(Float32, ns_x, ndo_x-1, N); # running cache for hidden states (used to generate observations)
    φ_t = zeros(Float32, ns_φ, ndo_φ, N); # running cache for observations

    dh_dr_self_array = zeros(Float32, D, ns_x, N); # running cache for partial derivatives of hidden states with respect to position of focal particle
    empty_sector_flags = falses(ns_x, ndo_x, N); # running cache for flags that indicate absence of observations for each modality & dyamical order & individual

    μ_x_n_t = zeros(Float32,ns_x*ndo_x) # running cache for storing instantaneous beliefs for a given individual

    perturbed_particle_idx = zeros(Int64,num_to_perturb,T_sim) # list of indices of which agents are getting perturbed

    external_r_hist = zeros(Float32, D, N, length(perturbation_times))

    perturb_time_count = 0

    for t = 2:T_sim # we start at 2nd timestep since we've already initialised the first timestep

        noise_samples_action = reshape(NoiseUtils.get_samples(noise_struct, D*N), (D, N))

        ####### UPDATE GENERATIVE PROCESS #######

        # update position using last timestep's velocity
        r[:,:,t] = r[:,:,t-1] .+ (dt .* v[:,:,t-1]) + (z_action .* noise_samples_action);
        r_t = @view r[:,:,t];

        v_past = @view v[:,:,t-1];
        v[:,:,t] = copy(v[:,:,t-1]); # carry forward last timestep's velocity to the current timestep

        # calculate distance between all particles
        # [dist_matrix[i,j] = norm(r_t[:,i] .- r_t[:,j]) for i = 1:N, j = 1:N]
        pairwise!(dist_matrix, Euclidean(), r_t, dims=2); # this goes way faster than the above approach

        noise_samples_φ = reshape(NoiseUtils.get_samples(noise_struct, ns_φ*ndo_φ*N), (ns_φ, ndo_φ, N))

        if t in perturbation_times

            perturb_time_count += 1

            d_group, _, _, relative_rankings = AnalysisUtils.compute_Dgroup_and_rankings_single(r[:,:,t],v[:,:,t])
            # the individuals we perturb are the ones at the very front of the school
            perturbed_particle_idx[1:num_to_perturb, t] = convert(Array{Int64,1},relative_rankings[1:num_to_perturb])


        else

            perturbed_particle_idx[1:num_to_perturb, t] .= 0

        end


        for n = 1:N # loop over individuals

            ####### get hidden states and observations #######

            neighbor_idx = findall(1:N .!== n) # all the particles besides me

            @views my_r, my_v = r_t[:,n], v_past[:,n] # focal particle's position and velocity

            @views other_r, other_v  = r_t[:,neighbor_idx], v_past[:,neighbor_idx] # all other particles' positions and velocities

            sector_idx = GeoUtils.assign_visfield_ids(my_r, my_v, other_r, ns_x, rotation_matrices, dist_thr); # returns a vector of BitArrays, one for each visual sector, that index whether neighbor particles are in the corresponding visual sector

            # calculate hidden states
            x_t[:,:,n], dh_dr_self_array[:,:,n], empty_sector_flags[:,:,n] =  SimUtils.calculate_sector_hidden_states(ns_x, ndo_x, D, sector_idx, dist_matrix[n,:], neighbor_idx, other_r, other_v, my_r, my_v, x_hist[:,:,n,t-1])

            # now modify the hidden state calculations using the 'external_position' if we are dealing with a perturbed particle

            if n in perturbed_particle_idx[:,t]

                displacement_vector = my_v .+ randn(D)
                external_r = reshape(my_r .+ (0.25f0 .* displacement_vector), D, 1)
                external_v = - 2f0 .* my_v # the 'external vector' is pointing in the opposite direction of the school

                external_r_hist[:, n, perturb_time_count] = external_r

                perturb_in_fields = GeoUtils.assign_visfield_ids(my_r, my_v, external_r, ns_x, rotation_matrices, dist_thr)

                which_ns = findfirst(vcat(perturb_in_fields...))

                if which_ns != nothing

                    if any(sector_idx[which_ns])
                        old_denom = Float32(sum(sector_idx[which_ns])) # we only do a running average version if there were other neighbors in that sector
                        new_denom = old_denom + 1f0
                    else
                        old_denom = 1f0
                        new_denom = 1f0
                    end

                    @views relative_external_r = external_r .- my_r
                    dist_x = norm(relative_external_r)
                    x_t[which_ns,1,n] = (x_t[which_ns,1,n] * old_denom + dist_x) / new_denom; # this is just the formula for a running average

                    @views relative_external_r_norm = relative_external_r ./ dist_x
                    external_dh_dr_self = -relative_external_r_norm; # partial derivative of total distance with respect to my position vector (unnormalized)
                    external_dh_dr_others = relative_external_r_norm; # partial derivative of total distance with respect to position of external particle (unnormalized)

                    old_total = x_t[which_ns,2,n] * old_denom
                    new_increment = sum(external_dh_dr_self .* my_v) + sum(external_dh_dr_others .* external_v);
                    x_t[which_ns,2,n] = (old_total + new_increment) / new_denom;

                    dh_dr_self_array[:,which_ns,n] = (dh_dr_self_array[:,which_ns,n] * old_denom .+ external_dh_dr_self) ./ new_denom;

                end
            end

            # generate observations from hidden states
            @views hidden_states_n, noise_n = x_t[:,:,n], noise_samples_φ[:,:,n]
            φ_t[:,:,n] = SimUtils.get_observations(ns_φ, ndo_φ, g, ∂g∂x, hidden_states_n, z_gp, noise_n)

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
                μ_x_n_t[(ns_φ*ndo_φ +1):end] .= 0f0 # do this to make sure acceleration beliefs from agent n-1  aren't 'shared' to agent n
            else
                μ_x_n_t = copy(μ_x[:,n,t-1]); # otherwise, carry forward optimised beliefs from the last timestep to be your initial beliefs for this timestep
            end

            μ_x[:,n,t], ε_z = SimUtils.run_belief_updating(μ_x_n_t, vectorized_φ, dt, num_iter, κ_μ,
                                                 ns_x, ndo_x, ns_φ, ndo_φ,
                                                 g, ∂g∂x, f, ∂f∂x, 𝚷_z, 𝚷_ω, D_shift)


            ####### Update actions using a gradient descent on free energy #######

            # dF/dv = dF/dphi * dphi/dv -- only non-zero term in the vector
            # dphi/dv = dphi'/dv -- i.e. the h-velocity observation (h_dot). So we only
            # need to use the prediction error related to the observation of
            # the h-velocity
            ∂F∂φprime = ε_z[(ns_φ+1):(ns_φ*ndo_φ)]; # the second ns_φ elements of the vector of sensory prediction errors correspond to the different elements of ∂F∂φprime

            # ∂φprime_∂v = ∂g∂x_μ2' .* dh_dr_self_array[:,:,n];
            # ∂φprime_∂v = dh_dr_self_array[:,:,n];

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

        end # end of loop over individuals

        v[:,:,t] = v[:,:,t] ./ sqrt.(sum(v[:,:,t].^2,dims=1)); # normalize velocities
        x_hist[:,:,:,t] .= copy(x_t)
        φ_hist[:,:,:,t] .= copy(φ_t)

    end # end of loop over time

    results_dict = Dict(:r => r, :v => v, :x_hist => x_hist, :φ_hist => φ_hist, :μ_x => μ_x, :dF_dv_hist => dF_dv_hist,
                        :external_r_hist => external_r_hist, :perturbed_particle_idx => perturbed_particle_idx)

    return results_dict

end # end of run_simulation function

end
