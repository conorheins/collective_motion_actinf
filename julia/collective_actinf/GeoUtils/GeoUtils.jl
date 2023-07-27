module GeoUtils

using LinearAlgebra, LazyGrids
using Distances
using Statistics

function rot_matrix(angle::Float32)::Matrix{Float32}
    return [cosd(angle) -sind(angle); sind(angle) cosd(angle) ]
end

function generate_start_end_rotation_lists(sector_angles::Array{Float32,1}; reverse_flag = false)::Tuple{Vector{Matrix}, Vector{Matrix}}
    """
    This function generates an array of arrays of arrays of rotation matrices corresponding to the rotations required to generate
    the sector-boundary vectors corresponding to the edges of visual zones. This is accomplished multiplying each matrix
    with the heading-direction vector of a particle, whose heading vector is assumed to have angle 0s (parallel to the axis of rotation)
    """

    R_starts = Vector{Matrix}(undef, length(sector_angles)-1);
    R_ends = Vector{Matrix}(undef, length(sector_angles)-1);

    if reverse_flag
        rotation = [0f0 -1f0; 1f0 0f0]
    else
        rotation = [1f0 0f0; 0f0 1f0] # otherwise just dot the rotation matrix with the identity matrix to yield itself
    end

    for s_i = 1:length(sector_angles)-1
        st_a, end_a = sector_angles[s_i], sector_angles[s_i+1]
        R_starts[s_i] = rotation'rot_matrix(st_a)
        R_ends[s_i] =  rotation'rot_matrix(end_a)
    end


    return R_starts, R_ends

end

function generate_hexagonal_grid(D::Int64, N_X::Int64, N_Y::Int64; hex_radius::Float32=1f0)::Array{Float32, 2}
    """
    This function initialises an array of hexagonally-spaced positions `r` for
    a multi-particle simulation. The spacing of the hexagonals (and thus the side length) is determined by a user-given desired hexagonal radius
    (distance between center of each hexagon and a vertex of the hexagon)
    INPUTS:
        `D`::Int64 - number of spatial dimensions (only 2 supported for now)
        `N_X`::Int64 - number of grid points in X (horizontal) direction
        `N_Y`::Int64 - number of grid points in Y (vertical) direction
        `hex_radius`::Float32 - optional parameter that gives the desired distance between the center of the hexagon and each vertex
    OUTPUTS:
        `r`::Array{Float32,2} - a size (D, N) array that stores the spatial positions of all the particles. r[1,:] is X positions, r[2,:] is Y positions.
    """

    N = N_X * N_Y # total number of individuals

    w = sqrt(3f0) * hex_radius # the spacing between the x-coordinates of each hexagonal center
    h = 2f0 * hex_radius # the spacing between the y-coordinates of each hexagonal center

    x_offset = 0.5f0 * w # the spacing between the x-coordinates of each hexagonal vertex
    y_offset = 0.25f0 * h # the spacing between the y-coordinates of each hexagonal vertex

    xv::Matrix{Float32}, yv::Matrix{Float32} = ndgrid(LinRange(x_offset, x_offset * N_X, N_X), LinRange(y_offset, y_offset * N_Y, N_Y))

    xv[:, 1:2:end] = xv[:, 1:2:end] .+ y_offset # stagger every 2nd x-coordinate to make a hexagon

    return vcat(vec(xv)', vec(yv)')
end

function filter_with_ellipse(coordinates::Array{Float32, 2}; rx = nothing, ry = nothing)::Array{Float32, 2}
    """
    Finds points within an ellipse centered at the center of mass of a set of coordinates, with optional horizontal and vertical
    axes of lengths `2 * rx` and `2 * ry`. If unspecified, `rx` is 1/4 the horizontal extent of the coordinates in the X axis,
    and `ry` is 1/2 the vertical extent of the coordinates in the Y axis
    """

    centroid = mean(coordinates, dims = 2)

    if isnothing(rx)
        rx = 0.25f0 .* (maximum(coordinates[1,:]) -  minimum(coordinates[1,:]))
    end

    if isnothing(ry)
        ry =  0.5f0 .* (maximum(coordinates[2,:]) -  minimum(coordinates[2,:]))
    end

    centered_scaled_x = ((coordinates[1,:] .- centroid[1]).^2) ./ (rx^2)
    centered_scaled_y = ((coordinates[2,:] .- centroid[2]).^2) ./ (ry^2)
    within_ellipse = (centered_scaled_x .+ centered_scaled_y) .< 1f0

    return coordinates[:,within_ellipse]

end

function initialize_velocities(D::Int64, N::Int64, mean_vel::Vector{Float32}, var_vel::Vector{Float32}; sampling_fn::Function = rand)::Array{Float32, 2}
    """
    Initializes velocity array of size (D, N) with a mean velocity vector `mean_vel`, a variance vector `var_vel` and a random sampling function
    `sampling_fn`, which is only constrained to take as arguments the dimensions of the returned array of random numbers.
    """
    v = reshape(mean_vel , D, 1) .+ (var_vel .* sampling_fn(Float32,D,N))

    v = v ./ sqrt.(sum(v.^2,dims=1)); # normalize velocities to unit length

    return v

end



function assign_visual_field_ids_new(R_starts, R_ends, r, v, dist_thr::Float32, distance_matrix)::Matrix{BitArray}
    """
    Vectorized version of the algorithm for assigning different neighbours to particular sensory sectors
    INPUTS:
        `R_starts`        : a `num_sectors` length vector containing rotation matrices (one for each sensory sector) that rotates an individual's velocity to the normal vector to the START (i.e. most CW) vector
                            of the respective sector
        `R_ends`          : a `num_sectors` length vector containing rotation matrices (one for each sensory sector) that rotates an individual's velocity to the normal vector to the END (i.e. most CCW) vector
                            of the respective sector
        `r`               : (D, N) matrix of the position vectors of all N agents
        `v`:              : (D, N) matrix of all the unit-normed velocity vectors of all N agents
        `dist_thr`        : the distance cut-off for inclusion in a sensory sector
        `distance_matrix` : (N, N) matrix of pairwise distances
    OUTPUTS:
        A matrix of size (N, num_sectors), where each entry (n, s) is a N-length BitArray whose binary elements indicate whether the agent with that index is within the s-th sensory sector of the n-th individual
    """
    all_sector_start_vectors_orth = map(x -> x'v, R_starts)
    all_sector_end_vectors_orth = map(x -> x'v, R_ends)

    relative_pos_vectors::Vector{Matrix{Float32}} = map(x -> r .- x, collect(eachcol(r)))

    all_CW_matrices_start::Vector{Matrix{Float32}} = map(x->compute_clockwise_dot_products(relative_pos_vectors, x), all_sector_start_vectors_orth)
    all_CW_matrices_end::Vector{Matrix{Float32}}  = map(x->compute_clockwise_dot_products(relative_pos_vectors, x), all_sector_end_vectors_orth)

    within_sector_CC::Vector{BitMatrix}  = map(check_clockwise_conditions, all_CW_matrices_start, all_CW_matrices_end)

    within_sector_assignments = map(x -> matrix_and(x, (distance_matrix .<= dist_thr)), within_sector_CC)

    return hcat(collect_per_column.(within_sector_assignments)...)

end

function compute_clockwise_dot_products(r_c::Vector{Matrix{Float32}}, reversed_sector_vectors::Matrix{Float32})::Matrix{Float32}
    return hcat(map(my_dot, r_c, collect(eachcol(reversed_sector_vectors)))...)
end

function my_dot(x::Matrix{Float32},y)::Array{Float32,1}
    return x'y
end

function check_clockwise_conditions(CW_matrix_start::Matrix{Float32}, CW_matrix_end::Matrix{Float32})::BitMatrix
    """
    Checks:
    1) whether the dot product results in CW_matrix_start are greater than or equal to 0 (and thus the corresponding coordinate is counterclockwise
    to the sector START vector
    2) whether the dot product results in CW_matrix_end are less than 0 (and thus the corresponding coordinate is clockwise
    to the sector END vector
    """
    return (CW_matrix_start .>= 0) .& (CW_matrix_end .< 0)
end

function matrix_and(input_matrix::BitMatrix, within_dist::BitMatrix)
    """
    Bitwise AND of two BitMatrix matrices
    """
    return input_matrix .& within_dist
end

function collect_per_column(input_matrix::BitMatrix)
    return collect(eachcol(input_matrix))
end

function h_per_sector(focal_dists, sector_idx_row)::Vector{Float32}
    """
    computes the average distance within each sensory sector, for a given vector of distances between each other
    agent and a given focal individual `focal_dists` and a vector of sector assignments from the perspective
    of that focal individual `sector_idx_row`
    """

    return map(x -> sum(focal_dists[x])./sum(x), sector_idx_row)

end


function get_all_sector_dists(dist_matrix, sector_idx::Matrix{BitArray})
    """
    Calculate the Euclidean sector-wise distances between each agent and the neighbours among
    the different sensory sectors
    """

    return map(h_per_sector, eachrow(dist_matrix), eachrow(sector_idx))

end


function norm_columns(input_matrix::Matrix{Float32})::Matrix{Float32}
    return input_matrix ./ sqrt.(sum(input_matrix.^2,dims=1))
end

function get_normed_sector_r_single(focal_r, r, sector_idx_row)::Vector{Matrix{Float32}}

    return norm_columns.(map(x -> r[:,x] .- focal_r, sector_idx_row))
end


function get_normed_sector_r_all(r, sector_idx)::Vector{Vector{Matrix{Float32}}}

    get_normed_fixed_r = (focal_r, sector_idx_row) -> get_normed_sector_r_single(focal_r, r, sector_idx_row)

    return map(get_normed_fixed_r, eachcol(r), eachrow(sector_idx))
end


function get_dh_dr_self_all(all_sector_vectors::Vector{Vector{Matrix{Float32}}})

    return map(x -> -mean.(x,dims=2), all_sector_vectors)

end

function get_dh_dr_others_all(all_sector_vectors::Vector{Vector{Matrix{Float32}}}, sector_idx::Matrix{BitArray})

    return map( (x,y) -> x ./ sum.(y), all_sector_vectors, eachrow(sector_idx))
end

function get_sector_v_single(v, sector_idx_row)
    return map(x -> v[:,x], sector_idx_row)
end

function get_sector_v_all(v, sector_idx)

    return map(x -> get_sector_v_single(v, x), eachrow(sector_idx))

end

function compute_self_velocity(focal_v, dh_dr_self)
    return map(x -> dot(focal_v, x), dh_dr_self)
end

function compute_other_velocity(dh_dr_other, sector_v)

    return map((x,y) -> sum(x .* y), dh_dr_other, sector_v)

end

function compute_hprime(all_dh_dr_self, v, all_dh_dr_others, all_sector_v)

    self_component = map((x, y) -> compute_self_velocity(x, y), eachcol(v), all_dh_dr_self)

    other_component = map(compute_other_velocity, all_dh_dr_others, all_sector_v)

    return self_component .+ other_component

end


nansum(x) = sum(filter(!isnan,x))

nansum(x,d) = mapslices(nansum,x,dims=d)

function nandot(x, y)
    return nansum(x' .* hcat(y...), 2)
end


function zero_out_nans!(vector::Vector{Float32})

    vector[isnan.(vector)] .= 0f0

    return vector

end

function calculate_sector_hidden_states_vectorized(dist_matrix, sector_idx, r, v)

    all_sector_h = zero_out_nans!.(get_all_sector_dists(dist_matrix, sector_idx))
    all_sector_r = get_normed_sector_r_all(r, sector_idx)

    all_dh_dr_self = get_dh_dr_self_all(all_sector_r)
    all_dh_dr_others = get_dh_dr_others_all(all_sector_r, sector_idx)

    all_sector_v = get_sector_v_all(v, sector_idx)

    all_sector_hprime = zero_out_nans!.(compute_hprime(all_dh_dr_self, v, all_dh_dr_others, all_sector_v))

    return all_sector_h, all_sector_hprime, all_dh_dr_self

end

function get_heading_vectors(positions::Array{Float32,2}, target::Array{Float32,2})
    """
    This returns the normalized vectors pointing from each D-dimensional coordinate stored in the columns of `positions`
    towards the target (a column vector of size (D,1), stored in `target`
    Arguments:
    =========
        `positions`     [Array{Float32,2}]: array of size (D, N) containing the D-dimensional position vectors of N individual particles
        `target`        [Array{Float32,2}]: array of size (D,1) containing the position of a single target
    Returns:
    ========
        `heading_vectors` [Array{Float32,2}]: normalized (i.e. unit) vectors pointing from each position vector stored in `positions` towards the target (`target`)
    """
    heading_vectors = mapslices(x -> x ./ norm(x), positions .- target, dims = 1)

    return heading_vectors

end

function calculate_dists_to_targets(r, v, preference_idx::Vector{Vector{Int64}}, target_pos::Vector{Array{Float32,2}})
    """
    Computes the hidden states (distance & velocity of distance) corresponding to the target hidden states
    Arguments:
    =========
        `r_t`                 [Array{Float32, 2}]: array of size (D, N) containing the position coordinates of all the particles
        `v_t`                 [Array{Float32, 2}]: array of size (D, N) containing the velocity coordinates of all the particles
        `preference_idx`      [Vector{Vector{Int64}}]: array whose elements correspond to the indices of the individuals that sense & are drawn towards ('prefer') each target
        `target_pos`          [Vector{Array{Float32,2}}]: array of size (num_targets, ). Each element of this array contains a (D, 1)-size vector which describe the Euclidean coordinates of each target - in column vector form.
    Returns:
    =========
        `tdist_per_target`      [Vector{Array{Float32,2}}]: vector of two column vectors - each column vector stores the distances between the lists of individuals in by `preference_idx` and the corresponding target in `target_pos`
        `tdist_prime_per_target`[Vector{Array{Float32,2}}]: vector of two row vectors - each row vector stores the time derivatives of the distances between the lists of individuals in by `preference_idx` and the corresponding target in `target_pos`
        `∂tdist∂r_per_target`   [Vector{Array{Float32,2}}]: vector of two matrices - each matrix stores the partial derivatives (each column in the matrix is partial derivatives of one individual) of the target-distance time-derivative with respect to the positions of each individual in the target-relative list
    """

    tdist_per_target = dropdims.(map( (idx, t_pos) -> pairwise(Euclidean(), r[:,idx], t_pos, dims = 2), preference_idx, target_pos), dims = 2)

    ∂tdist∂r_per_target = map( (idx, t_pos) -> get_heading_vectors(r[:,idx], t_pos), preference_idx, target_pos)

    tdist_prime_per_target = dropdims.(map( (idx, ∂tdist∂r) -> sum(v[:,idx] .* ∂tdist∂r,dims=1), preference_idx, ∂tdist∂r_per_target), dims = 1)

    return tdist_per_target, tdist_prime_per_target, ∂tdist∂r_per_target
end


function assign_visfield_ids(my_r, my_vel, other_r, num_sectors, rotation_matrices, dist_thr)
    """
    This function assigns the positions of other particles (indices given by the columns of `other_r`) to membership to
    one of `num_sectors` visual field sectors relative to position/velocity of a focal particle (given by `my_r`/`my_vel` respectively).
    This is achieved by first computing the vectors that form the boundaries of each visual field sector (using `rotation_matrices` to
    rotate `my_vel` to each of these sector boundaries), and then the logical intersection of 3 simple conditions to determine whether
    each position is within a given sector -- whether the position is 'counterclockwise' to the start of the sector, whether the position is 'clockwise'
    to the end of the sector, and whether the position is within `dist_thr` distance units from `my_r`.
    INPUTS:
        `my_r`::Array{Float32,1} - size (D,) position vector of the focal particle
        `my_vel`::Array{Float32,1}
        `other_r`::Array{Float32,2}
        `num_sectors`::Int64
        `rotation_matrices`::Array{Array{Real,2},1} - rotation matrices for the different sectors
        `dist_thr`::Float32
    OUTPUTS:
        `sector_assignments`::Vector{BitArray{1}} - logical vectors that store assignments of each neighbor to the different visual sectors, going
          from right to left.
    """

    # sector_vectors = Vector{Array{Float32,2}}(undef, num_sectors);
    #
    # for s_i = 1:num_sectors
    #     sector_vectors[s_i] = hcat(rotation_matrices[s_i]*my_vel, rotation_matrices[s_i+1]*my_vel)
    #     # sector_vectors[s_i][:,1] contains the most clockwise vector edge of the visual sector (the 'start sector')
    #     # sector_vectors[s_i][:,2] contains the most counterclockwise vector edge of the visual sector (the 'end sector')
    # end
    #
    # sector_assignments = Vector{Array{Bool,1}}(undef, num_sectors)
    #
    # for s_i = 1:num_sectors
    #     sector_assignments[s_i] = check_inside_sector(other_r, my_r, sector_vectors[s_i][:,1],sector_vectors[s_i][:,2],dist_thr);
    # end
    #
    # return sector_assignments

    # faster alternative (but you don't save the sector vectors)

    sector_assignments = Vector{BitArray{1}}(undef, num_sectors)

    # REMEMBER, YOU DINGUS - `START_SECTOR_VECTOR` IS TO THE RIGHT OF `END_SECTOR_VECTOR`, i.e. IT"s CLOCKWISE OF END_SECTOR_VECTOR.
    # THE FIRST SECTOR IS ALWAYS THE FURTHEST CLOCKWISE (RIGHT SIDE OF THE PARTICLE) -- LAST SECTOR IS THE FURTHEST 'COUNTERCLOCKWISE' (LEFT SIDE OF PARTICLE)
    for s_i = 1:(num_sectors)
        start_sector_vector, end_sector_vector = rotation_matrices[s_i]'my_vel, rotation_matrices[s_i+1]'my_vel
        sector_assignments[s_i] = check_inside_sector(other_r, my_r, start_sector_vector, end_sector_vector, dist_thr);
    end

    return sector_assignments

end

function id_particles_at_angles(r_t::Array{Float32, 2},v_t::Array{Float32, 2}, rot_angle_vec::Array{Float32, 1})::Array{Int64,1}

    """
    This function identifies individuals closest to a vector of provided angles, where the angles are with respect to the heading direction of the group.
    INPUTS:
        `r_t`::Array{Float32,2} - size (D,N) matrix of position vectors of all particles
        `v_t`::Array{Float32, 2} - size (D,N) matrix of heading vectors of all particles
        `rot_angle_vec`::Array{Float32,1} - vector of angles at which to id individuals
    OUTPUTS:
        `individuals_at_angles`::Array{Int64,1} - vector of particle ids, with one id corresponding to the individual whose relative position is the closest to the corresponding entry of `rot_angle_vec`
    """

    # group heading direction
    group_vel_t = sum(v_t, dims = 2) ./ size(v_t, 2);
    group_vel_t = vec(group_vel_t ./ sqrt.(sum(group_vel_t.^2,dims=1)))

    # center relative coordinates of each individual
    group_pos = sum(r_t,dims=2) ./ size(r_t, 2);
    r_c_t = r_t .- group_pos;

    # projects them onto a unit-circle around the group center
    r_c_t ./= sqrt.(sum(r_c_t.^2,dims=1))

    individuals_at_angles::Array{Int64, 1} = zeros(Int64, length(rot_angle_vec))

    for (r_i, rot_angle) in enumerate(rot_angle_vec)
        R_matrix = [cosd(rot_angle) -sind(rot_angle); sind(rot_angle) cosd(rot_angle) ]
        rotated_group_vel = R_matrix'group_vel_t
        ranked_indiv = sortperm(vec(rotated_group_vel'r_c_t), rev = true)

        idx_counter = 1
        closest_indiv = ranked_indiv[idx_counter]
        while closest_indiv in individuals_at_angles
            idx_counter += 1
            closest_indiv = ranked_indiv[idx_counter]
        end

        individuals_at_angles[r_i] = closest_indiv
    end

    return individuals_at_angles
end

function check_inside_sector(neighbor_pos, my_pos, sector_start, sector_end, dist_thr)

    relative_pos = neighbor_pos .- my_pos; # center the positions on the focal agent's position

    ids = .!are_clockwise(sector_start, relative_pos) .& are_clockwise(sector_end, relative_pos) .& is_within_radius(relative_pos, dist_thr.^2);

    return ids

end

function are_clockwise(sector_start, relative_pos)

    normal_vector = [-1, 1] .* reverse(sector_start)

    return (relative_pos'normal_vector .< 0f0); # BitArray of true/false Boolean indicators telling you whether the relativized neighbor position is clockwise to the sector vector

    # See https://stackoverflow.com/questions/13652518/efficiently-find-points-inside-a-circle-sector:
    # "If the projection is a positive number, then the v2 is positioned counter-clockwise to v1. Otherwise, v2 is clockwise to v1."
end

function is_within_radius(relative_pos, radius_squared)

    return vec(sum(relative_pos.^2,dims=1) .<= radius_squared)

end

function distance_calc1(positions::Array{Float32,3})

    D, N, T = size(positions)
    dist_matrix::Array{Float32,2} = zeros(Float32,N,N)

    for t = 1:T
        pairwise!(dist_matrix, Euclidean(), positions[:,:,t], dims=2);
    end

    return dist_matrix

end

function distance_calc2(positions::Array{Float32,3})

    D, N, T = size(positions)
    dist_matrix::Array{Float32,2} = zeros(Float32,N,N)

    for t = 1:T
        [dist_matrix[i,j] = norm(positions[:,i,t] .- positions[:,j,t]) for i = 1:N, j = 1:N]
    end

    return dist_matrix

end

end
