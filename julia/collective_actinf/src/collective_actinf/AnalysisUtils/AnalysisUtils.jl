module AnalysisUtils

using Distances
using NearestNeighbors
using Statistics
using StatsBase
using LsqFit
using LinearAlgebra
using LightGraphs
using JLD
using DSP

function compute_euclidean_distance(coordinate_matrix::Array{Float32,2})::Array{Float32,2}
    """
    Given a `coordinate matrix` of shape D x N (where D is the number of spatial dimensions,
    N is the number of particles / individuals), computes the Euclidean distances between all the coordinates
    using the pairwise() function from the Distances package. The resulting dist_matrix
    is therefore of size N x N.
    """

    dist_matrix = pairwise(Euclidean(), coordinate_matrix, dims = 2)

    return dist_matrix

end

function compute_tdist_N(coordinate_matrix::Array{Float32,2},target_position::Array{Float32,2})::Array{Float32,2}
    """
    Given a `coordinate matrix` of shape D x N (where D is the number of spatial dimensions,
    N is the number of particles / individuals) and `target_position` of shape D x 1, this function
    computes the Euclidean distances between all the coordinates and `target_position` using the
    pairwise() function from the Distances package. The resulting t_distances
    is of size N x 1
    """

    t_distances = pairwise(Euclidean(), coordinate_matrix, target_position, dims = 2)

    return t_distances

end

function compute_NN_distances_over_time(positions::Array{Float32,3})::Array{Float32,3}
    """
    Given a time series of `positions` (of shape D x N x T, where D is the number of spatial dimensions,
    N is the number of particles / individuals, and T is the number of time points), compute a time-series
    of the pairwise Euclidean between the particles at each time point. The resulting dist_matrix_t
    is therefore of size N x N x T.
    """

    dist_matrix_t = mapslices(compute_euclidean_distance, positions, dims = [1,2])

    return dist_matrix_t

end

function extract_lower_triangular_elements(input_matrix::Array{Float32,2})::Array{Float32,1}
    """
    Extract lower triangle of a matrix into a vector
    """
    return input_matrix[tril!(trues(size(input_matrix)), -1)]
end

function compute_summary_distance_over_time(dist_matrix_t::Array{Float32,3}; statistic::String="mean")::Array{Float32,1}
    """
    Given a time series of distance matrices stores in `dist_matrix_t` of size
    N x N x T, compute a summary statistic of the pairwise distances at each time step.
    The lower triangle of each distance matrix is first extracted using
    """

    N = Float32(size(dist_matrix_t,1))

    lower_triangle_dists = mapslices(extract_lower_triangular_elements,dist_matrix_t,dims=[1,2])
    lower_triangle_dists = dropdims(lower_triangle_dists,dims=2)

    if statistic == "mean"
        output = mean(lower_triangle_dists,dims=1)
    elseif statistic == "median"
        output = median(lower_triangle_dists,dims=1)
    end

    return vec(output)
end

function compute_group_polarization(velocities::Array{Float32,3})::Array{Float32,1}
    """
    This function calculates the time-dependent polarization of an ensemble of particles, whose
    velocities are stored in the array velocities of shape D x N x T (D = spatial dimensionality, N = number of particles,
    T = number of timesteps)
    """

    N = size(velocities,2)

    p_group::Array{Float32,1} = dropdims(sqrt.(sum(sum(velocities,dims=2).^2,dims=1)),dims=(1,2)) ./ N

    return p_group

end

function compute_angular_momentum(positions::Array{Float32,3},velocities::Array{Float32,3})::Array{Float32,1}
    """
    This computes the group averaged angular momentum over time using a definition of cross product for
    2-D vectors defined as cross(x,y) = x1*y2 - x2*y1 (a scalar output)
    """
    n_particles::Int64 = size(positions,2);

    # calculate group centroid
    c_group::Array{Float32,3} = sum(positions,dims=2) ./ n_particles;

    # calculate every fish's position relative to the group centroid, over time
    r_c::Array{Float32,3} = positions .- c_group;
    r_c = r_c ./ sqrt.(sum(r_c.^2,dims=1)) # normalize it to unit length

    cross_products::Array{Float32,2} = (velocities[1,:,:] .* r_c[2,:,:]) .- (velocities[2,:,:] .* r_c[1,:,:])

    M::Array{Float32,1} = vec(abs.(sum(cross_products,dims=1)./n_particles))

    return M

end

function compute_angular_momentum_cross(positions::Array{Float32,3},velocities::Array{Float32,3})::Array{Float32,1}
    """
    This computes the group averaged angular momentum over time using the Julia implementation of cross product.
    Loops over particle id and time to perform the computation.
    """
    n_particles::Int64 = size(positions,2);

    # calculate group centroid
    c_group::Array{Float32,3} = sum(positions,dims=2) ./ n_particles;

    # calculate every fish's position relative to the group centroid, over time
    r_c::Array{Float32,3} = positions .- c_group;
    r_c = r_c ./ sqrt.(sum(r_c.^2,dims=1)) # normalize it to unit length

    @views augmented_r = vcat(r_c, zeros(Float32,1,n_particles,size(positions,3)))
    @views augmented_v = vcat(velocities, zeros(Float32,1,n_particles,size(velocities,3)))

    cross_products::Array{Float32,3} = zeros(Float32,3,n_particles,size(positions,3))
    for t = 1:size(positions,3)
        for n = 1:n_particles
            cross_products[:,n,t] = cross(augmented_v[:,n,t], augmented_r[:,n,t])
        end
    end

    sum_across_individuals::Array{Float32,3} = sum(cross_products,dims=2)

    M::Array{Float32,1} = vec(sqrt.(sum(sum_across_individuals.^2,dims=1))./n_particles)

    return M

end

function compute_Dgroup_and_rankings_single(positions::Array{Float32,2},velocities::Array{Float32,2})::Tuple{Array{Float32,1},Array{Float32,1},Array{Float32,1},Array{Float32,1}}
    """
    This function computes the heading direction of the entire group (`d_group`), the centroid (`c_group`), the
    angular distances between every fish and the heading-direction weighted center of the group. In this version of the function
    it is computed for a single timestep.
    """

    n_particles::Int64 = size(positions,2);

    # calculate group centroid
    c_group::Array{Float32,2} = sum(positions,dims=2) ./ n_particles;

    # compute group heading direction and normalize to unit length
    d_group::Array{Float32,2} = sum(velocities, dims = 2) ./ n_particles;
    d_group ./= sqrt.(sum(d_group.^2,dims=1))

    # calculate every fish's position relative to the group centroid
    r_c::Array{Float32,2} = positions .- c_group;

    relative_dists::Array{Float32,1}  = vec(d_group'r_c)

    relative_rankings::Array{Float32,1} = sortperm(relative_dists,rev=true);

    return vec(d_group), vec(c_group), relative_dists, relative_rankings

end

function compute_Dgroup_and_rankings(positions::Array{Float32,3},velocities::Array{Float32,3})::Tuple{Array{Float32,2},Array{Float32,2},Array{Float32,2},Array{Float32,2}}
    """
    This function computes the heading direction of the entire group (`d_group`), the centroid (`c_group`), the
    angular distances between every fish and the heading-direction weighted center of the group. In this version of the function
    it is computed for a whole timeseries of positions and velocities.
    """

    n_particles::Int64 = size(positions,2);

    # calculate group centroid
    c_group::Array{Float32,3} = sum(positions,dims=2) ./ n_particles;

    # compute group heading direction and normalize to unit length
    d_group::Array{Float32,3} = sum(velocities, dims = 2) ./ n_particles;
    d_group ./= sqrt.(sum(d_group.^2,dims=1))

    # calculate every fish's position relative to the group centroid
    r_c::Array{Float32,3} = positions .- c_group;

    relative_dists::Array{Float32,2}  = dropdims(sum(r_c .* d_group,dims=1),dims=1);

    relative_rankings::Array{Float32,2} = zeros(Float32,size(relative_dists));

    for t = 1:size(relative_rankings,2)
    	relative_rankings[:,t] = sortperm(relative_dists[:,t],rev=true);
    end

    return dropdims(d_group,dims=2), dropdims(c_group,dims=2), relative_dists, relative_rankings

end

function compute_elongation_t(positions::Array{Float32,3},velocities::Array{Float32,3})
    """
    This function uses a simple method to identify two perpendicular axes of the school. The elongation
    metric is the ratio of the length of the axis that is parallel to the average velocity vector (`d_group`)
    to the length of the 'second' principal axis, which is orthogonal to the principal axis.
    If elongation >> 1, this means the school is moving with an elongated shape, whereas elongation << 1 means the
    school is 'fatter' than it is 'long', with respect to the directional axis of group movement.
    """

    D::Int64, n_particles::Int64, T::Int64 = size(positions)

    # calculate group centroid (mean)
    c_group::Array{Float32,3} = sum(positions,dims=2) ./ n_particles;

    # compute group heading direction and normalize to unit length
    d_group::Array{Float32,2} = dropdims(sum(velocities, dims = 2) ./ n_particles, dims=2)
    d_group ./= sqrt.(sum(d_group.^2,dims=1))

    # calculate vectors normal to group heading direction
    rotation_matrix::Array{Float32,2} = [0f0 -1f0; 1f0 0f0] # 90 degree rotation matrix

    rotated_d_group::Array{Float32,2} = mapslices(x -> rotation_matrix*x,d_group, dims=1) # this rotates the d_group vector at every time by 90 degrees

    # calculate every fish's position relative to the group centroid, and made the N count be in the first dimension
    r_c::Array{Float32,3} = positions .- c_group

    d_group_projections::Array{Float32,2} = dropdims(sum(r_c .* reshape(d_group, D, 1, T),dims=1),dims=1)
    sort!(d_group_projections, dims=1, rev=true)

    rot_d_group_projections::Array{Float32,2} = dropdims(sum(r_c .* reshape(rotated_d_group, D, 1, T),dims=1),dims=1)
    sort!(rot_d_group_projections, dims=1, rev=true)

    @views c_group_dimdropped = dropdims(c_group,dims=2)

    front_points = c_group_dimdropped .+ (reshape(d_group_projections[1,:],1,T) .* d_group)
    back_points = c_group_dimdropped .+ (reshape(d_group_projections[end,:],1,T)  .* d_group)

    d_group_axis_lengths = sqrt.(sum( (front_points .- back_points).^2, dims = 1))

    front_points = c_group_dimdropped .+ (reshape(rot_d_group_projections[1,:],1,T) .* rotated_d_group)
    back_points = c_group_dimdropped .+ (reshape(rot_d_group_projections[end,:],1,T) .* rotated_d_group)

    rot_d_group_axis_lengths = sqrt.(sum( (front_points .- back_points).^2, dims = 1))

    return vec(d_group_axis_lengths ./ rot_d_group_axis_lengths)

end


function is_connected_over_time(positions::Array{Float32,3};threshold::Float32 = 5f0)::BitArray{1}
    """
    Given a time series of `positions` (of shape D x N x T, where D is the number of spatial dimensions,
    N is the number of particles / individuals, and T is the number of time points), compute a time-series
    of Boolean (true/false) indicators of whether the group is cohesive over time.
    Approach: compute the time-dependent pairwise Euclidean between the particles at each time point using the function
    `compute_NN_distances_over_time` The resulting sequence of nearest-neighbor distances is then thresholded using
    threshold to create a series of binary, symmetric adjacency matrices at each timestep. The LightGraphs.jl
    package is used to convert these adjacency matrices into an array of SimpleGraph types. Then the is_connected() function
    from LightGraphs is then broadcasted across the graph-array to return the BitArray indicating whether the graph is connected
    (read: "the group is connected") at each timestep
    """

    distance_matrices = compute_NN_distances_over_time(positions)

    adj_mat_sequence = convert(Array{Int64,3},distance_matrices .< threshold)

    graphs = dropdims(mapslices(Graph,adj_mat_sequence,dims=[1,2]),dims=(1,2))

    connected_over_time = broadcast(is_connected,graphs)

    return connected_over_time

end

function compute_knn_over_time(positions::Array{Float32,3}; k::Int64 = 5, statistic::String="mean")::Array{Float32,1}
    """
    Given a time series of `positions` of shape (D, N, T) where D is number of dimensions, N is the
    number of particles, and T is the timesteps, and `k` which is an integer indicating the number of nearest neighbors
    to compute, compute the history of nearest neighbor distances over time for each particle. For each timestep, the mean
    of the nearest neighbor distances per particle is computed, and then the statistic parameter (either "mean" or "median")
    is used to compute a summary of the particle-specific means, either by the average or median across particles. The time-series
    of this statistic is then returned in `knn_over_time`
    """

    kdtrees = dropdims(mapslices(KDTree, positions, dims=[1,2]),dims=(1,2)) # create a nearest neighbor tree for each time slice of positions

    # find the k-nearest neighbors for each time slice

    dists = [knn(kdtrees[t],positions[:,:,t],k+1)[2] for t = 1:size(positions,3)]

    if statistic == "mean"
        knn_over_time = mean.([(sum.(dist_i)./k) for dist_i in dists])
    elseif statistic == "median"
        knn_over_time = median.([(sum.(dist_i)./k) for dist_i in dists])
    end

    return knn_over_time

end

function compute_tdist_over_time(positions::Array{Float32,3}, target_position::Array{Float32,2})::Array{Float32,2}
    """
    Given a time series of `positions` (of shape D x N x T, where D is the number of spatial dimensions,
    N is the number of particles / individuals, and T is the number of time points), and a `target_position`
    of size (D,1) storing the position of the target, compute a time-series
    of the Euclidean distances between each particle and the target at each time point. The resulting tdist_history
    is therefore of size N x T;
    """

    # this returns a N x 1 x T array, so we need to just squeeze out the middle singleton dimension
    tdist_over_time = mapslices(x -> compute_tdist_N(x,target_position), positions, dims=[1,2])

    # dropping the singleton dimension in the middle
    tdist_over_time = dropdims(tdist_over_time,dims=2)

    return tdist_over_time

end

function get_decision_latency(r::Array{Float32,3}, target_pos::Vector{Array{Float32,2}}; tdist_thr::Float32 = 1f0)
    """
    Logic is to find the latency at which the school reached the first target, and use that as the decision latency for the trial
    """

    T_max = size(r,3)
    is_close(x) = findfirst(x .<= tdist_thr)

    target_latencies = zeros(Int64,length(target_pos))
    for target_i in 1:length(target_pos)
        tdists_over_time = compute_tdist_over_time(r, target_pos[target_i])

        first_arrivals = is_close.(eachrow(tdists_over_time))
        first_arrivals = ifelse.(isnothing.(first_arrivals), T_max, first_arrivals)
        target_latencies[target_i] = minimum(first_arrivals)
    end

    if all(target_latencies .== T_max)
        return (T_max, NaN)
    else
        return findmin(target_latencies)
    end

end

function compute_angle_to_target(point_hist::Array{Float32,2}, target_pos::Array{Float32,1}; origin::Array{Float32,1} = [0f0, 0f0])
    """
    Compute angle subtended by each point in some point history `point_hist`, relative to some target position `target_pos`

    Arguments:
    =========
        `point_hist` [Array{Float32,2}]: array of size (D, T) that contains a list of D-dimensional coordinates over time
        `target_pos` [Array{Float32,1}]: vector of size (D,), the coordinates of the target against which the angle will be measured
        `origin`     [Array{Float32,1}, optional]: origin that determines the corner of the triangle opposite to the side spanned by each
                    point in `point_hist` and the point described by `target_pos`
    Returns:
    =========
        `angles`   [Array{Float32,1}]: array of size (T,) that contains the angles over time
    """
    target_vector = target_pos .- origin # generate the target vector
    sq_dist = dot(target_vector, target_vector) # get the magnitude of the target vector

    scales = sum(point_hist .* target_vector,dims=1) ./ sq_dist; # this gives you the scalar that encodes the position along the line `target_vector`
    projected_point_hist = scales .* target_vector

    # use acos(adjacent / hypotenuse) (the 'CAH' from SOHCAHTOA) to calculate the angle
    adjacent_lengths = mapslices(x -> norm(x), projected_point_hist, dims = 1)
    hypot_lengths = mapslices(x -> norm(x), point_hist, dims = 1)

    AH = clamp.(adjacent_lengths ./ hypot_lengths, -1f0, 1f0)
    angles = acosd.(AH)

    return vec(angles)
end

function compute_angle_to_target_2(point_hist::Array{Float32,2}, target_pos::Array{Float32,1})
    """
    Compute angle subtended by each point in some point history `point_hist`, relative to some target position `target_pos`

    Arguments:
    =========
        `point_hist` [Array{Float32,2}]: array of size (D, T) that contains a list of D-dimensional coordinates over time
        `target_pos` [Array{Float32,1}]: vector of size (D,), the coordinates of the target against which the angle will be measured
        `origin`     [Array{Float32,1}, optional]: origin that determines the corner of the triangle opposite to the side spanned by each
                    point in `point_hist` and the point described by `target_pos`
    Returns:
    =========
        `angles`   [Array{Float32,1}]: array of size (T,) that contains the angles over time
    """
    target_vector = target_pos ./ norm(target_pos)# generate the target vector

    normed_point_hist = mapslices(x -> x./ norm(x), point_hist, dims = 1)

    all_dot_products = clamp.(sum(normed_point_hist .* target_vector, dims = 1), -1f0, 1f0)
    angles = acosd.(all_dot_products)

    return vec(angles)
end

function compute_batch_cross_product(reference_vectors, compare_vectors)

    v1x_v2y = selectdim(reference_vectors, 1, 1) .* selectdim(compare_vectors, 1, 2)
    v1y_v2x = selectdim(reference_vectors, 1, 2) .* selectdim(compare_vectors, 1, 1)
    turning_magnitudes = v1x_v2y .- v1y_v2x # if this is > 0, then group turned left, if this is < 0, then group turned right

    return turning_magnitudes

end


function generate_param_configs(param1, param2...)::Array{Float64,2}

    # create the matrix of parameter configurations (one configuration per row)
    all_param_configs = Iterators.product(param1, param2...)
    all_param_configs = mapreduce(x -> hcat(x...), vcat, all_param_configs)

    return all_param_configs

end
function analyze_data(results_folder::String, param_configs, n_trials::Int64; threshold::Float32=10f0, window_size::Int64 = 1000, pos_var_name::String = "r", v_var_name::String = "v")::Array{Float32,3}
    """
    Function for analyzing free schooling behavior from simulation results, where data is stored as history of positions, history of velocities, and the condition.
        Results array is stored in a 3-D matrix, where rows indicate the condition index, columns indicate different dependent measures, and 3-D 'slice' indexes the sufficient statistic (mean or SEM)
    """

    num_configs, num_conditions = size(param_configs)

    results_array::Array{Float32,3} = zeros(Float32, num_configs, 10, 2) # 10 dependent measures we're interested in; two slices (one for MEAN, one for SEM)

    first_fname = readdir(joinpath(results_folder,"1"))[1]
    trial_fname_prefix = first_fname[1:findfirst(isequal('_'), first_fname)]
    # findfirst(isequal(::Char), ::String) returns the first index of where ::Char is found in string
    # this line basically assumes the first file of [results_folder]/1 will look like a typical .jld file that stores
    # the trial-specific data (r, v, etc.), so we take the first few characters (including the '_') of that filename
    # as our template for how the trial-specific data files are stored

    for config_i in 1:num_configs

        condition_folder::String = joinpath(results_folder,string(config_i));

        # load the first trial of the condition in order to find the condition index in the scheme of all conditions (won't necessarily match up)
        config = collect(load(joinpath(condition_folder,string(trial_fname_prefix,"1.jld")), "condition"))
        param_config_idx = findfirst(all(param_configs[:,1:num_conditions] .== reshape(config,(1,num_conditions)),dims=2))[1]

        mean_NN_dist_all_trials::Array{Float32,1} = zeros(Float32,n_trials)
        var_NN_dist_all_trials::Array{Float32,1} = zeros(Float32,n_trials)

        mean_polar_all_trials::Array{Float32,1} = zeros(Float32,n_trials)
        var_polar_all_trials::Array{Float32,1} = zeros(Float32,n_trials)

        mean_connected_all_trials::Array{Float32,1} = zeros(Float32,n_trials)

        mean_angular_momentum_all_trials::Array{Float32,1} = zeros(Float32,n_trials)
        var_angular_momentum_all_trials::Array{Float32,1} = zeros(Float32,n_trials)

        milling_flags::Array{Float32,1} = zeros(Float32,n_trials)

        mean_elongation_all_trials::Array{Float32,1} = zeros(Float32,n_trials)
        var_elongation_all_trials::Array{Float32,1} = zeros(Float32,n_trials)

        for trial_i = 1:n_trials

            positions, velocities = load(joinpath(condition_folder, string(trial_fname_prefix,trial_i,".jld")),pos_var_name,v_var_name)

            T_max = size(positions,3)

            if size(positions, 2) <= 5
                dist_matrix_t::Array{Float32,3} = compute_NN_distances_over_time(positions[:,:,(T_max-window_size):T_max])
                knn_over_time::Array{Float32,1}  = compute_summary_distance_over_time(dist_matrix_t)
            else
                knn_over_time = compute_knn_over_time(positions[:,:,(T_max-window_size):T_max],k=5,statistic="median")
            end

            mean_NN_dist_all_trials[trial_i] = mean(knn_over_time);
            var_NN_dist_all_trials[trial_i]  = var(knn_over_time);

            # polarization
            polar_over_time::Array{Float32,1} = compute_group_polarization(velocities[:,:,(T_max-window_size):T_max])

            mean_polar_all_trials[trial_i] = mean(polar_over_time)
            var_polar_all_trials[trial_i] = var(polar_over_time)

            # connectedness
            connected_over_time::BitArray{1} = is_connected_over_time(positions[:,:,(T_max-window_size):T_max],threshold=threshold)

            mean_connected_all_trials[trial_i] = mean(connected_over_time)

            # time-dependent computation of angular momentum
            angular_momentum_over_time::Array{Float32,1} = compute_angular_momentum(positions[:,:,(T_max-window_size):T_max], velocities[:,:,(T_max-window_size):T_max])

            mean_angular_momentum_all_trials[trial_i] = mean(angular_momentum_over_time)
            var_angular_momentum_all_trials[trial_i] = var(angular_momentum_over_time)

            milling_flags[trial_i] = (mean(angular_momentum_over_time) > 0.5f0) ? 1f0 : 0f0

            # time-dependent computation of elongation
            elongation_over_time::Array{Float32,1} = compute_elongation_t(positions[:,:,(T_max-window_size):T_max], velocities[:,:,(T_max-window_size):T_max])
            mean_elongation_all_trials[trial_i] = mean(elongation_over_time)
            var_elongation_all_trials[trial_i] = var(elongation_over_time)

        end

        sem_c::Float32 = Float32(sqrt(n_trials))
        #first slice is for means
        results_array[param_config_idx,:,1] = [mean(mean_NN_dist_all_trials), mean(var_NN_dist_all_trials), mean(mean_polar_all_trials), mean(var_polar_all_trials), mean(mean_connected_all_trials),
                                               mean(mean_angular_momentum_all_trials), mean(var_angular_momentum_all_trials), mean(milling_flags), mean(mean_elongation_all_trials), mean(var_elongation_all_trials)]

        # second slice is for SEM
        results_array[param_config_idx,:,2] = (1f0 ./ sem_c) .* [std(mean_NN_dist_all_trials), std(var_NN_dist_all_trials), std(mean_polar_all_trials), std(var_polar_all_trials), std(mean_connected_all_trials),
                                                std(mean_angular_momentum_all_trials), std(var_angular_momentum_all_trials), 0f0, std(mean_elongation_all_trials), std(var_elongation_all_trials)]

    end

    save(joinpath(results_folder,"results.jld"), "results_array", results_array,"n_trials",n_trials)

    return results_array

end

function analyze_milling_probability(results_folder::String, param_configs, n_trials::Int64; window_size::Int64 = 1000)::Array{Float32,1}
    num_configs, num_conditions = size(param_configs)

    results_array::Array{Float32,1} = zeros(Float32, num_configs)

    for config_i in 1:num_configs

        condition_folder::String = joinpath(results_folder,string(config_i));

        # load the first trial of the condition in order to find the condition index in the scheme of all conditions (won't necessarily match up)
        config = collect(load(joinpath(condition_folder,"Trial_1.jld"), "condition"))
        param_config_idx = findfirst(all(param_configs[:,1:num_conditions] .== reshape(config,(1,num_conditions)),dims=2))[1]

        milling_flags::Array{Float32,1} = zeros(Float32,n_trials)

        for trial_i = 1:n_trials

            positions, velocities = load(joinpath(condition_folder, string("Trial_",trial_i,".jld")),"r","v")

            T_max = size(positions,3)

            # time-dependent computation of angular momentum
            angular_momentum_over_time::Array{Float32,1} = compute_angular_momentum(positions[:,:,(T_max-window_size):T_max], velocities[:,:,(T_max-window_size):T_max])

            milling_flags[trial_i] = (mean(angular_momentum_over_time) > 0.5f0) ? 1f0 : 0f0

        end

        # compute the milling probability as the proportion of trials (for this condition) that ended up with milling groups
        results_array[param_config_idx] = mean(milling_flags)

    end

    return results_array
end

function fit_t_constants(results_folder::String, param_configs, n_trials::Int64, autocorr_lags::AbstractArray{T,1} where T<:Integer; start_idx::Int64 = 1)::Array{Float32,2}
    """
    Function for analyzing free schooling behavior from simulation results, where data is stored as history of positions, history of velocities, and the condition.
        Results array is stored in a 3-D matrix, where rows indicate the condition index, columns indicate different dependent measures, and 3-D 'slice' indexes the sufficient statistic (mean or SEM)
    """

    num_configs, num_conditions = size(param_configs)

    t_constants = zeros(Float32, num_configs, n_trials) # store all the fitted time constants

    p0 = [0.5] # initial (pre-fitting) time constant

    @. model(x,p) = exp(-x*p[1]) # specify the single exponential decay model that you're fitting

    for config_i in 1:num_configs

        condition_folder::String = joinpath(results_folder,string(config_i));

        # load the first trial of the condition in order to find the condition index in the scheme of all conditions (won't necessarily match up)
        config = collect(load(joinpath(condition_folder,"Trial_1.jld"), "condition"))
        param_config_idx = findfirst(all(param_configs[:,1:num_conditions] .== reshape(config,(1,num_conditions)),dims=2))[1]

        tau_all_trials = zeros(Float32,n_trials)

        for trial_i = 1:n_trials

            v = load(joinpath(condition_folder, string("Trial_",trial_i,".jld")),"v")

            v = v[:,:,start_idx:end]

            v_all = dropdims(sum(v,dims=2),dims=2)
            v_all = v_all ./ sqrt.(sum(v_all.^2,dims=1)) # normalize velocity vector to unit norm
            summed_v = sum(v_all,dims=1) # add up x and y velocities

            autocorr_y = autocor(vec(summed_v),autocorr_lags) # calculate the ydata for the model by using the empirical autocorrelation function on the summed velocity vector

            exponential_fit = curve_fit(model, autocorr_lags, autocorr_y, p0)

            tau_all_trials[trial_i] = Float32(coef(exponential_fit)[1])

        end

        t_constants[config_i,:] = copy(tau_all_trials)

    end

    save(joinpath(results_folder,"autocorr_results.jld"), "t_constants", t_constants)

    return t_constants

end

function compute_residual_variance(results_folder::String, param_configs, n_trials::Int64; win_len::Int64 = 10, start_idx::Int64 = 1)

    num_configs, num_conditions = size(param_configs)

    error_var = zeros(Float32, num_configs, n_trials) # store all the fitted time constants

    conv_window = ones(Float32,win_len) # a box-car moving window
    norm_factor = sum(conv_window) # this is used to normalize the convolution results

    cutoff = Int64(win_len/2)

    for config_i in 1:num_configs

        condition_folder::String = joinpath(results_folder,string(config_i));

        # load the first trial of the condition in order to find the condition index in the scheme of all conditions (won't necessarily match up)
        config = collect(load(joinpath(condition_folder,"Trial_1.jld"), "condition"))
        param_config_idx = findfirst(all(param_configs[:,1:num_conditions] .== reshape(config,(1,num_conditions)),dims=2))[1]

        error_var_trials = zeros(Float32,n_trials)

        for trial_i = 1:n_trials

            v = load(joinpath(condition_folder, string("Trial_",trial_i,".jld")),"v")

            v = v[:,:,start_idx:end]

            D, N, T = size(v)

            all_particles_error_var = zeros(Float32,N)

            smoothed_vel = zeros(Float32,D,T) # cache for storing the smoothed velocity

            for n = 1:N

                @views v_n = v[:,n,:]
                for d_i = 1:D
                    smoothed_vel[d_i,:] = conv(conv_window,v_n[d_i,:])[cutoff:(end-cutoff)] ./ norm_factor
                end

                error_var_n = sum((v_n .- smoothed_vel).^2, dims = 2)./ Float32(T) # time averaged squared deviation from the mean (i.e. average variance, with moving average correction)
                all_particles_error_var[n] = mean(error_var_n) # average across spatial dimensions

            end

            error_var_trials[trial_i] = mean(all_particles_error_var)

        end

        error_var[config_i,:] = copy(error_var_trials)

    end

    save(joinpath(results_folder,"error_var_results.jld"), "error_var", error_var)

    return error_var

end


function recompute_rotation_order(results_folder::String, param_configs::Array{Float64,2}, n_trials::Int64; window_size::Int64 = 1000)::Array{Float32,3}
    """
    Function for recomputing the rotational order stuff and replacing existing results_array with corrected version, since original implementation was wrong (forgot to normalise the centroid-relative position vectors to unit norm, `r_c`)
    """

    num_configs, num_conditions = size(param_configs)

    results_array::Array{Float32,3} = load(joinpath(results_folder,"results.jld"),"results_array") # re-load the old results array

    for config_i in 1:num_configs

        condition_folder::String = joinpath(results_folder,string(config_i));

        # load the first trial of the condition in order to find the condition index in the scheme of all conditions (won't necessarily match up)
        config = collect(load(joinpath(condition_folder,"Trial_1.jld"), "condition"))
        param_config_idx = findfirst(all(param_configs[:,1:num_conditions] .== reshape(config,(1,num_conditions)),dims=2))[1]

        mean_angular_momentum_all_trials::Array{Float32,1} = zeros(Float32,n_trials)
        var_angular_momentum_all_trials::Array{Float32,1} = zeros(Float32,n_trials)

        milling_flags::Array{Float32,1} = zeros(Float32,n_trials)

        for trial_i = 1:n_trials

            positions, velocities = load(joinpath(condition_folder, string("Trial_",trial_i,".jld")),"r","v")

            T_max = size(positions,3)

            # time-dependent computation of angular momentum
            angular_momentum_over_time::Array{Float32,1} = compute_angular_momentum(positions[:,:,(T_max-window_size):T_max], velocities[:,:,(T_max-window_size):T_max])

            mean_angular_momentum_all_trials[trial_i] = mean(angular_momentum_over_time)
            var_angular_momentum_all_trials[trial_i] = var(angular_momentum_over_time)

            milling_flags[trial_i] = (mean(angular_momentum_over_time) > 0.5f0) ? 1f0 : 0f0

        end

        sem_c::Float32 = Float32(sqrt(n_trials))

        #first slice is for means
        results_array[param_config_idx,6:8,1] = [mean(mean_angular_momentum_all_trials), mean(var_angular_momentum_all_trials), mean(milling_flags)]

        # second slice is for SEM
        results_array[param_config_idx,6:8,2] = (1f0 ./ sem_c) .* [std(mean_angular_momentum_all_trials), std(var_angular_momentum_all_trials), 0f0]


    end

    save(joinpath(results_folder,"results.jld"), "results_array", results_array,"n_trials",n_trials)

    return results_array


end

function compute_average_elongation(results_folder::String, param_configs, n_trials::Int64; window_size::Int64 = 1000)::Array{Float32,2}

    num_configs, num_conditions = size(param_configs)

    mean_elongation::Array{Float32,2} = zeros(Float32, num_configs, n_trials) # store all the fitted time constants

    for config_i in 1:num_configs

        condition_folder::String = joinpath(results_folder,string(config_i));

        # load the first trial of the condition in order to find the condition index in the scheme of all conditions (won't necessarily match up)
        config = collect(load(joinpath(condition_folder,"Trial_1.jld"), "condition"))
        param_config_idx = findfirst(all(param_configs[:,1:num_conditions] .== reshape(config,(1,num_conditions)),dims=2))[1]

        elongation_trials = zeros(Float32,n_trials)

        for trial_i = 1:n_trials

            r, v = load(joinpath(condition_folder, string("Trial_",trial_i,".jld")),"r","v")
            T_max = size(r,3)

            elongation_over_time = compute_elongation_t(r[:,:,(T_max-window_size):T_max],v[:,:,(T_max-window_size):T_max])

            elongation_trials[trial_i] = mean(elongation_over_time)

        end

        mean_elongation[config_i,:] = copy(elongation_trials)

    end

    return mean_elongation

end

function compute_perturbation_histograms(r_all::Array{Float32,4}, v_all::Array{Float32,4}; x_bin_edges::Array{Float32,1} = [-5f0, 5f0], y_bin_edges::Array{Float32,1} = [-5f0 5f0], Δx::Float32 = 1f0)::Array{Float32,4}

    # D, N, realisation_length, num_realisations  = size(perturbation_results[:r_all]) # old version, used up for scripts multimodal_LS_perturbations[1-4].jl
    D, N, realisation_length, num_realisations  = size(r_all)

    bin_edges = (collect(Float32,x_bin_edges[1]:Δx:x_bin_edges[2]), collect(Float32,y_bin_edges[1]:Δx:y_bin_edges[2]))
    all_hists::Array{Float32,4} = zeros(Float32,length(bin_edges[1])-1,length(bin_edges[2])-1,realisation_length,num_realisations)

    for real_i = 1:num_realisations

        r_realisation = r_all[:,:,:,real_i]
        v_realisation = v_all[:,:,:,real_i]

        d_group, c_group, _, _ = compute_Dgroup_and_rankings(r_realisation,v_realisation)

        # we first want to orient the group heading direction to be parallel to the vertical axis (the y-axis)
        rotation_angle = acos(d_group[2,1]) # this is the angle between the initial heading direction of the group and the y-axis ([0 1]) - shorthand-way to compute acos([0, 1] * d_group[:,1])
        if d_group[1,1] < 0f0 # if the heading direction is pointing to the left, we want to do a CW rotation
            rotation_matrix = [cos(rotation_angle) sin(rotation_angle); -sin(rotation_angle) cos(rotation_angle)]
        elseif d_group[1,1] > 0f0 # if the heading direction is pointing to the right, we want to do a CCW rotation
            rotation_matrix = [cos(rotation_angle) -sin(rotation_angle); sin(rotation_angle) cos(rotation_angle)]
        end

        # aligns the average heading direction of the group at the first timestep with the y_axis
        aligned_d_group = rotation_matrix*d_group[:,1]

        # this vector is the directional heading of the perturbed particles (since we defined them to switch exactly to the heading that is normal to the average motion vector)
        perturbed_vector = [0f0 -1f0; 1f0 0f0]*aligned_d_group[:,1]

        # aligns all individual velocities with y axis
        aligned_v = mapslices(x -> rotation_matrix*x, v_realisation, dims=(1,2))

        # this measures the 'correlation' (cosine of the angle) between each particle's motion vector and the perturbed vector
        all_vector_products = dropdims(mapslices(x -> perturbed_vector'x, aligned_v, dims=(1,2)), dims = 1)

        # center all positions on point (0,0)
        centered_r = r_realisation .- reshape(c_group, D, 1, size(c_group,2))

        for t = 1:(realisation_length-1) # subtract 1 since the first timestep is the timestep immediately prior to the perturbation

            h_t_counts = fit(Histogram, (centered_r[1,:,t+1], centered_r[2,:,t+1]), bin_edges)
            h_t_weighted = fit(Histogram, (centered_r[1,:,t+1], centered_r[2,:,t+1]), StatsBase.weights(all_vector_products[:,t+1]), bin_edges)

            all_hists[:,:,t,real_i] = h_t_weighted.weights ./ h_t_counts.weights

        end

    end

    all_hists[isnan.(all_hists)] .=0f0

    return all_hists
end

function calculate_mean_perturbation_histograms(results_folder, num_configs, n_initialisations)::Tuple{Array{Float32,4}, Array{Float32,4}}

    example_hist = load(joinpath(results_folder,"1/init_1.jld"),"all_hists")

    num_x_bins, num_y_bins, realisation_length, n_realisations = size(example_hist)

    mean_per_condition = zeros(Float32, num_x_bins, num_y_bins, realisation_length-1, num_configs)
    std_per_condition = zeros(Float32, num_x_bins, num_y_bins, realisation_length-1, num_configs)

    for config_i in 1:num_configs

    	condition_folder = joinpath(results_folder,string(config_i));

    	for init_i = 1:n_initialisations

    		all_hists = load(joinpath(condition_folder,string("init_",init_i,".jld")), "all_hists")

    		mean_across_real = dropdims(mean(all_hists[:,:,2:end,:],dims=4),dims=4)
            std_across_real  = dropdims(std(all_hists[:,:,2:end,:],dims=4),dims=4)

    		mean_per_condition[:,:,:,config_i] += (mean_across_real ./ Float32(n_initialisations))
            std_per_condition[:,:,:,config_i] += (std_across_real ./ Float32(n_initialisations)) # what you'll end up with is average standard deviation across initializations, where std is computed across realizations, per initialization

    	end

    end

    return mean_per_condition, std_per_condition

end

function calculate_mean_hists_from_raw(results_folder, num_configs, n_initialisations; x_bin_edges::Array{Float32,1} = [-5f0, 5f0], y_bin_edges::Array{Float32,1} = [-5f0 5f0], Δx::Float32 = 1f0)::Tuple{Array{Float32,4}, Array{Float32,4}}

    example_r_all = load(joinpath(results_folder,"1/init_1.jld"),"r_all")

    _, _, realisation_length, n_realisations = size(example_r_all)

    num_x_bins, num_y_bins = length(x_bin_edges[1]:Δx:x_bin_edges[2]), length(y_bin_edges[1]:Δx:y_bin_edges[2])

    mean_per_condition = zeros(Float32, num_x_bins-1, num_y_bins-1, realisation_length-1, num_configs)
    std_per_condition = zeros(Float32, num_x_bins-1, num_y_bins-1, realisation_length-1, num_configs)

    for config_i in 1:num_configs

    	condition_folder = joinpath(results_folder,string(config_i));

    	for init_i = 1:n_initialisations

            r_all, v_all = load(joinpath(condition_folder,string("init_",init_i,".jld")), "r_all", "v_all")

            all_hists = compute_perturbation_histograms(r_all, v_all, x_bin_edges = x_bin_edges, y_bin_edges = y_bin_edges, Δx = Δx)

    		mean_across_real = dropdims(mean(all_hists[:,:,2:end,:],dims=4),dims=4)
            std_across_real  = dropdims(std(all_hists[:,:,2:end,:],dims=4),dims=4)

    		mean_per_condition[:,:,:,config_i] += (mean_across_real ./ Float32(n_initialisations))
            std_per_condition[:,:,:,config_i] += (std_across_real ./ Float32(n_initialisations)) # what you'll end up with is average standard deviation across initializations, where std is computed across realizations, per initialization

    	end

    end

    return mean_per_condition, std_per_condition

end


function get_perturbed_nonperturbed_idx(r_perturbations::Array{Float32, 4}, v_perturbations::Array{Float32,4}, num_perturbed::Int64)::Tuple{Array{Int64,1}, Array{Int64,1}, Array{Float32,1}}

    _, _, front_back_scores, relative_rankings = compute_Dgroup_and_rankings_single(r_perturbations[:,:,1,1],v_perturbations[:,:,1,1]) # the perturbed/non-perturbed indices will be the same for all realisations, so just pick the first one
    perturbed_idx = convert(Array{Int64,1},relative_rankings[1:num_perturbed]) # the first `num_perturbed` ones will be the indices of those that were perturbed, by design of the experiment
    non_perturbed_idx = convert(Array{Int64,1},relative_rankings[num_perturbed+1:end]) # anyone left are the non-perturbed

    return perturbed_idx, non_perturbed_idx, front_back_scores

end

function compute_relative_angle_changes(v_perturbations::Array{Float32,4}, particle_ids::Array{Int64,1})::Array{Float32,2}

    initial_velocity_self = v_perturbations[:,particle_ids,1,:][:,:,[CartesianIndex()],:]
    self_dot_products = clamp.(dropdims(sum(initial_velocity_self .* v_perturbations[:,particle_ids,:,:], dims=1),dims=1), -1f0, 1f0)
    all_angles =  dropdims(mean(acosd.(self_dot_products),dims=3),dims=3)

    return all_angles

end

function compute_front_back_correlation(front_back_scores::Array{Float32,1}, all_angles::Array{Float32,2}, particle_ids::Array{Int64,1})::Array{Float32,2}

    non_perturbed_FB = front_back_scores[particle_ids]
    rates_of_change = mean(diff(all_angles[:,1:50],dims=2),dims=2)
    return hcat([non_perturbed_FB, rates_of_change]...)

end

end
