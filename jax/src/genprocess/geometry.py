import jax.numpy as jnp
from jax import vmap, random
from jax_md import space
from jax_md.space import map_product, distance
import numpy as np
import networkx as nx

euclidean_dist_fn, _ = space.free()
euclidean_dist_vmapped = map_product(euclidean_dist_fn)

def compute_pairwise_dists(pos):

    return distance(euclidean_dist_vmapped(pos, pos))

def subtract(x, y):
    return x - y

def elementwise_matrix_mult(arr1, arr2):
    return arr1 * arr2

# def remove_nans(arr):
#     """ Replaces NaN-valued elements of an array with 0.0's """

#     return arr * jnp.logical_not(jnp.isnan(arr))

def remove_nans(arr):
    """ Replaces NaN-valued elements of an array with 0.0's """

    return jnp.nan_to_num(arr)

def initialize_positions_velocities(init_key, 
                                    N, 
                                    pos_x_bounds = [-1.0, 1.0], 
                                    pos_y_bounds = [-1.0, 1.0], 
                                    vel_x_bounds = [-1.0, 1.0],
                                    vel_y_bounds = [-1.0, 1.0]
                                    ):
    """ 
    Initializes N position and velocity vectors given some bounds on the two dimensions of each. Unit-normalizes
    velocity vectors 
    """

    pos_key, vel_key = random.split(init_key, 2)

    # initialize positions using given bounds
    pos_x = random.uniform(pos_key, minval=pos_x_bounds[0], maxval=pos_x_bounds[1], shape = (N, 1))
    _, pos_key = random.split(pos_key, 2)
    pos_y = random.uniform(pos_key, minval=pos_y_bounds[0], maxval=pos_y_bounds[1], shape = (N, 1))
    init_pos = jnp.hstack((pos_x, pos_y))

    # initialize velocities using given bounds and normalize at end
    vel_x = random.uniform(vel_key, minval=vel_x_bounds[0], maxval=vel_x_bounds[1], shape = (N, 1))
    _, vel_key = random.split(vel_key, 2)
    vel_y = random.uniform(vel_key, minval=vel_y_bounds[0], maxval=vel_y_bounds[1], shape = (N, 1))
    init_vel = normalize_array(jnp.hstack((vel_x, vel_y)), axis = 1)

    return init_pos, init_vel

def advance_positions(pos_past, vel_past, action_fluctuations, speed=1.0, dt=0.01):
    """ Advance array of positions using array of velocities and some action fluctuations
    Parameters
    ----------
    pos_past : jnp.ndarray of shape (N, 2)
        Array of positions at the previous timestep
    vel_past : jnp.ndarray of shape (N, 2)
        Array of velocities at the previous timestep
    action_fluctuations : jnp.ndarray of shape (N, 2)
        Array of action fluctuations at the current timestep
    dt : float, optional
        Timestep size, by default 0.01
    """  

    pos_current = pos_past + (speed * dt * vel_past) + action_fluctuations

    return pos_current

def normalize_array(array, axis = 1):
    """
    Normalize an array along some given axis so that it has unit normed vectors stored along that dimension
    """

    norms = jnp.sqrt((array**2).sum(axis = axis))
    reshape_dims = [(1 if ii == axis else dim) for ii, dim in enumerate(array.shape)]
    return jnp.divide(array, norms.reshape(reshape_dims))

def safe_normalize(x, *, p=2):
    """
    Safely project a vector onto the sphere wrt the ``p``-norm. This avoids the
    singularity at zero by mapping zero to the uniform unit vector proportional
    to ``[1, 1, ..., 1]``.

    :param numpy.ndarray x: A vector
    :param float p: The norm exponent, defaults to 2 i.e. the Euclidean norm.
    :returns: A normalized version ``x / ||x||_p``.
    :rtype: numpy.ndarray
    """
    assert isinstance(p, (float, int))
    assert p >= 0
    norm = jnp.linalg.norm(x, p, axis=-1, keepdims=True)
    x = x / jnp.clip(norm, a_min=jnp.finfo(x).tiny)
    # Avoid the singularity.
    mask = jnp.all(x == 0, axis=-1, keepdims=True)
    x = jnp.where(mask, x.shape[-1] ** (-1 / p), x)
    return x

def rotation_matrix(rad):
    """ Computes rotation matrix given an angle (in radians) """
    return jnp.array([ [jnp.cos(rad), -jnp.sin(rad)], 
                       [jnp.sin(rad), jnp.cos(rad)]  ] )    

def compute_nearest_neighbour_vectors(pos):
    """ 
    Compute the vectors pointing from each vector in `pos` to every other vector in `pos` 

    Arguments:
    =========
    `pos` [jnp.ndarray of shape (N, D)]: matrix of `N` position vectors, each defined by `D` coordinates

    Returns:
    =========
    `n2n_vecs` [jnp.ndarray of shape (N, N, D)]: tensor containing vectors pointing from each individual to each other individual, where each slice (in the axis = 2 dimension) stores
                                                one coordinate of the vector. So  n2n_vecs[i,j,d] encodes the d-th component of the vector pointing from agent_i to agent_j. In other
                                                words, `n2n_vecs[i,j] == pos[j] - pos[i]`
    """
    
    n2n_vecs = vmap(subtract, (None, 0), 0)(pos, pos)

    return n2n_vecs

def compute_rotation_matrices(sector_angles, reverse_flag = True):
    """
    This function generates an array of arrays of arrays of rotation matrices corresponding to the rotations required to generate
    the sector-boundary vectors corresponding to the edges of visual zones. This is accomplished multiplying each matrix
    with the heading-direction vector of a particle, whose heading vector is assumed to have angle 0s (parallel to the axis of rotation)
    """

    rotation = jnp.array([ [0., 1.], [-1., 0.]]) if reverse_flag else jnp.array([ [1., 0.], [0., 1.]])
    # rotation = jnp.array([ [1., 0.], [0., -1.]]) if reverse_flag else jnp.array([ [1., 0.], [0., 1.]])

    R_starts, R_ends = np.zeros( (len(sector_angles)-1, 2, 2) ), np.zeros( (len(sector_angles)-1, 2, 2) )
    for ii in range(len(sector_angles)-1):

        # convert angles to radians before passing into `rotation_matrix` function
        start_angle, end_angle = jnp.radians(sector_angles[ii]), jnp.radians(sector_angles[ii+1])

        R_starts[ii] = rotation @ rotation_matrix(start_angle) 
        R_ends[ii] =  rotation @ rotation_matrix(end_angle)

    return R_starts, R_ends

def compute_sector_vectors(vel, R_starts, R_ends):
    """
    Rotates each velocity according to the rotation matrices that bring the velocity parallel to the sector start/end vectors
    corresponding to each sector
    """

    start_vecs = vmap(jnp.dot, (None, 0), 0)(vel, R_starts)
    end_vecs = vmap(jnp.dot, (None, 0), 0)(vel, R_ends)

    return start_vecs, end_vecs

def compute_clockwise_proj_single(sector_vecs, n2n_vecs):
    """
    Arguments:
    =========
    `sector_vecs` [jnp.ndarray of shape (n_agents, D)]: matrix of rotated velocity vectors for `n_agents` that are aligned with a given sector's boundary (either start or end)
    `n2n_vecs` [jnp.ndarray of shape (N, N, D)]: tensor containing vectors pointing from each individual to each other individual, where each slice (in the axis = 2 dimension) stores
                                                one coordinate of the vector. So  n2n_vecs[i,j,d] encodes the d-th component of the vector pointing from agent_i to agent_j                                             
    """

    out = vmap(jnp.dot, (0, 0))(n2n_vecs, sector_vecs)

    return out

def compute_clockwise_proj_all(start_vecs, end_vecs, n2n_vecs):
    """
    Vmaps the calculation of clockwise/counterclockwise projection coefficient of all individuals for a given sector, 
    across all sectors, and does this for both the "start" and "end" boundaries of these sectors serially
    """

    cw_proj_start = vmap(compute_clockwise_proj_single, (0, None))(start_vecs, n2n_vecs)
    cw_proj_end = vmap(compute_clockwise_proj_single, (0, None))(end_vecs, n2n_vecs)

    return cw_proj_start, cw_proj_end

def compute_visual_neighbours(pos, vel, R_starts, R_ends, dist_thr):
    """ 
    Compute the neighbours of all agents according to presence in their visual sectors
    Arguments:
    =========
    `pos` [jnp.ndarray of shape (N, D)]: matrix of `N` position vectors, each defined by `D` coordinates (each row)
    `vel` [jnp.ndarray of shape (N, D)]: matrix of `N` velocity vectors, each defined by `D` coordinates (each row)
    `R_starts` [jnp.ndarray of shape (n_sectors, D, D)]: matrix of rotation matrices, where each R_starts[i] encodes the rotation matrix that brings the focal velocity vector of an agent
                                                        in alignment with the left-most boundary of the i-th sector (sector "start")
    `R_ends` [jnp.ndarray of shape (n_sectors, D, D)]: matrix of rotation matrices, where each R_ends[i] encodes the rotation matrix that brings the focal velocity vector of an agent
                                                        in alignment with the right-most boundary of the i-th sector (sector "end") 
    `dist_thr` [float]: threshold of distance within which neighbours are perceived    
    Returns:
    =========       
    `within_sector_idx` [jnp.ndarray of shape (n_sectors, N, N)]: tensor of flags indicating presence/absence of neighbours within different sectors,
                                                                where within_sector_idx[i, j, k] tells whether neighbour `k` is within sector `i` of focal agent `j`
    `distance_matrix` [jnp.ndarray of shape (N, N)]: matrix of pairwise Euclidean distances between each of the particles
    `n2n_vecs` [jnp.ndarray of shape (N, N, D)]: tensor containing vectors pointing from each individual to each other individual, where each slice (in the axis = 2 dimension) stores
                                                one coordinate of the vector. So  n2n_vecs[i,j,d] encodes the d-th component of the vector pointing from agent_i to agent_j
    """

    start_vecs, end_vecs = compute_sector_vectors(vel, R_starts, R_ends) 

    n2n_vecs = compute_nearest_neighbour_vectors(pos) 

    distance_matrix = jnp.linalg.norm(n2n_vecs, axis = 2)
    # distance_matrix = compute_pairwise_dists(pos)

    cw_proj_start, cw_proj_end = compute_clockwise_proj_all(start_vecs, end_vecs, n2n_vecs)

    within_sector_CC = (cw_proj_start >= 0.) & (cw_proj_end < 0.) # flags for whether each point is within the sector

    within_sector_idx = vmap(lambda x,y: x & y, (0, None), 0)(within_sector_CC, distance_matrix < dist_thr)

    return within_sector_idx, distance_matrix, n2n_vecs

def compute_h_per_sector(within_sector_idx, distance_matrix):
    """ Calculate average distance per sector, for all agents """

    n_neighbours = within_sector_idx.sum(axis=2)
    h_per_sector = (within_sector_idx * distance_matrix[None,...]).sum(axis=2) / n_neighbours

    return remove_nans(h_per_sector)

def compute_h_per_sector_keepNaNs(within_sector_idx, distance_matrix):
    """ Calculate average distance per sector, for all agents """

    n_neighbours = within_sector_idx.sum(axis=2)
    h_per_sector = (within_sector_idx * distance_matrix[None,...]).sum(axis=2) / n_neighbours

    return h_per_sector

def compute_hprime_vectorized(all_dh_dr_self, vel, all_dh_dr_others, sector_v):

    self_component = (all_dh_dr_self * vel[None,...]).sum(-1) # sum out the last axis, computing the dot product of each focal agent's velocity vector with its dh_dr_self vector
    other_component = (all_dh_dr_others * sector_v).sum(axis=(-1,-2)) # sum out the last two axes, computing the dot products of each j-th "sector-velocity" vector with the corresponding j-th `all_dh_dr_others` vector, and then summing all the dot products together over j (vectorized across sectors and individuals)

    return self_component + other_component

def compute_hprime_per_sector(within_sector_idx, pos, vel, n2n_vecs):
    """ Compute hprime, or the time derivative of the average distance per sector """

    expanded_wsect_idx = within_sector_idx[...,None] # add a lagging dimension to `within_sector_idx` to enable broadcasted multiplications
    sector_r = expanded_wsect_idx * n2n_vecs[None, ...] # matrix of shape (n_sectors, N, N, D) where sector_r[i, j, k, :] contains the "sector vector" pointing from neighbour k to focal agent j within sector i i.e. pos[k,:] - pos[j,:]

    # normalize all sector vectors to unit norm
    sector_r /= jnp.linalg.norm(sector_r, axis = 3, keepdims=True)
    sector_r = remove_nans(sector_r)

    all_dh_dr_others = sector_r / expanded_wsect_idx.sum(axis=2, keepdims = True)

    # you can compute `all_dh_dr_others` first and then using it, compute `all_dh_dr_self` as follows (need to test):
    all_dh_dr_self = -all_dh_dr_others.sum(axis=2) # gradient of the average sector-wise distance with respect to oneself

    # old way of computing
    # all_dh_dr_self =  -sector_r.sum(axis=2) / expanded_wsect_idx.sum(axis=2) # gradient of the average sector-wise distance with respect to oneself

    sector_v = expanded_wsect_idx * vel[None, None, ...] # matrix of shape (n_sectors, N, N, D) where sector_v[i, j, k, :] contains the "sector velocity" of neighbour k within sector `i` of focal agent j

    hprime = compute_hprime_vectorized(all_dh_dr_self, vel, all_dh_dr_others, sector_v)

    return remove_nans(hprime), all_dh_dr_self

def compute_hprime_per_sector_special(within_sector_idx, pos, vel, n2n_vecs):
    """ Compute hprime, or the time derivative of the average distance per sector """

    expanded_wsect_idx = within_sector_idx[...,None] # add a lagging dimension to `within_sector_idx` to enable broadcasted multiplications
    sector_r = expanded_wsect_idx * n2n_vecs[None, ...] # matrix of shape (n_sectors, N, N, D) where sector_r[i, j, k, :] contains the "sector vector" pointing from neighbour k to focal agent j within sector i i.e. pos[k,:] - pos[j,:]

    # normalize all sector vectors to unit norm
    sector_r /= jnp.linalg.norm(sector_r, axis = 3, keepdims=True)
    sector_r = remove_nans(sector_r)

    all_dh_dr_others = sector_r / expanded_wsect_idx.sum(axis=2, keepdims = True)

    # you can compute `all_dh_dr_others` first and then using it, compute `all_dh_dr_self` as follows (need to test):
    all_dh_dr_self = -all_dh_dr_others.sum(axis=2) # gradient of the average sector-wise distance with respect to oneself

    # old way of computing
    # all_dh_dr_self =  -sector_r.sum(axis=2) / expanded_wsect_idx.sum(axis=2) # gradient of the average sector-wise distance with respect to oneself

    sector_v = expanded_wsect_idx * vel[None, None, ...] # matrix of shape (n_sectors, N, N, D) where sector_v[i, j, k, :] contains the "sector velocity" of neighbour k within sector `i` of focal agent j

    hprime = compute_hprime_vectorized(remove_nans(all_dh_dr_self), vel, remove_nans(all_dh_dr_others), sector_v)

    return hprime, all_dh_dr_self

## OLD JULIA CODE
#     all_dh_dr_self = get_dh_dr_self_all(all_sector_r)
#     all_dh_dr_others = get_dh_dr_others_all(all_sector_r, sector_idx)

#     all_sector_v = get_sector_v_all(v, sector_idx)

#     all_sector_hprime = zero_out_nans!.(compute_hprime(all_dh_dr_self, v, all_dh_dr_others, all_sector_v))

def compute_dist_matrices_over_time(pos):
    """ Assumes `pos` has shape (T, N, spatial_dim) """

    displacement_fn, _ = space.free()

    vmapped_displacement_fn = map_product(displacement_fn)

    def compute_distance_matrix_t(pos_matrix):
        return distance(vmapped_displacement_fn(pos_matrix, pos_matrix))
    
    dist_matrices_t = vmap(compute_distance_matrix_t)(pos)

    return dist_matrices_t

def is_connected_over_time(pos_hist, thr=5.0):
    """ Function that computes a binary timeseries of 1s and 0s that indicate whether there is a single connected component
    in the graph of pairwise distances between the agents. This is used to determine whether the agents are "cohesive" or not.
    
    Parameters
    ----------
    pos_hist : ndarray
        A T x N x 2 array of agent positions
    thr : float (optional)
        The threshold distance for the pairwise distances, used to construct the graph from which connectedness will be inferred.
        Default is 5.0.
    """
    dist_matrices_ts = np.array(compute_dist_matrices_over_time(pos_hist))
    
    is_connected = [nx.is_connected(nx.from_numpy_array(dist_mat < thr)) for dist_mat in dist_matrices_ts]
   
    return is_connected

def compute_angular_momentum_t(pos_t, vel_t):
    """ Computes the angular momentum of the agents around their center of mass over time """

    # compute the centroid of the group
    c_group_t = compute_group_centroid(pos_t)

    # compute the centroid-relative position vectors of all agents in the group
    pos_c = pos_t - c_group_t[None,...]
    pos_c /= jnp.linalg.norm(pos_c, axis=1, keepdims=True)

    # compute the angular momentum of each agent's velocity vector around the group centroid
    ang_mom_t = jnp.cross(pos_c, vel_t, axis=1)

    # average across individuals and take the absolute value
    ang_mom_t = jnp.absolute(jnp.nanmean(ang_mom_t))

    return ang_mom_t

def compute_angular_momentum(pos_hist, vel_hist):
    return vmap(compute_angular_momentum_t)(pos_hist, vel_hist)

def compute_sector_dists_over_time(pos_hist, vel_hist, genproc):

    def compute_sector_h(pos, vel):

        # compute visual neighbourhoods 
        within_sector_idx, distance_matrix, _ = compute_visual_neighbours(pos, vel, genproc['R_starts'], genproc['R_ends'], genproc['dist_thr'])

        # get h (first order observations)
        h = compute_h_per_sector_keepNaNs(within_sector_idx, distance_matrix)

        return h

    return vmap(compute_sector_h, (0, 0))(pos_hist, vel_hist)

def compute_and_rectify_turning_magnitudes(reference_velocity, compare_velocities):
    """ 
    Computes turning angle over time compared to some reference velocity stored in `reference_velocity` and then
    if the turning angle for a particular agent is majority of the time (computed the first 1/4 of the perturbation length) negative, 
    then flip its sign
    """ 

    turning_angles_uncorrected = vmap(compute_cross_products, in_axes = (None, 0))(reference_velocity, compare_velocities)

    total_time = turning_angles_uncorrected.shape[0]
    # mean_turning_per_agent = turning_angles_uncorrected[:int(total_time * 0.25)].mean(axis=0) # this will then be a (N,) size vector
    
    # turning_angles_rectified = turning_angles_uncorrected * jnp.sign(mean_turning_per_agent)[None,...]

    # fraction_below_zero = (turning_angles_uncorrected < 0.).sum(axis=0) > int(total_time*0.5) # flag that says, for each agent, if more than half the time it's response was below 0
    idx_below_zero = jnp.where(jnp.sign(turning_angles_uncorrected[:int(total_time*0.25)]).sum(0) <= 0., -1., 1.)

    turning_angles_rectified = turning_angles_uncorrected * idx_below_zero[None,...]
    # turning_angles_rectified = turning_angles_uncorrected.at[:,fraction_below_zero].set(-1 * turning_angles_uncorrected[:,fraction_below_zero]) # flip the sign if the majority of the time the turn was below 0

    return turning_angles_rectified, turning_angles_uncorrected

def compute_turning_magnitudes(reference_velocity, compare_velocities):
    """ Vmap cross-product computataion `compute_cross_products` over the time dimension"""
    return vmap(compute_cross_products, in_axes = (None, 0))(reference_velocity, compare_velocities)

def compute_cross_products(reference_vec, compare_vec):
    """ 
    Computes cross product between a given vector `compare_vec` and some reference vector `reference_vec`
    Columns of each store spatial dimensions, so shapes of both vectors are (N, 2) 
    """ 
    v1x_v2y = reference_vec[:,0] * compare_vec[:,1] 
    v1y_v2x = reference_vec[:,1] * compare_vec[:,0]

    return v1x_v2y - v1y_v2x # if quantity is > 0, then the agents turned left; if quantity is < 0, then the agents turned right

def compute_group_heading_direction(vel_t):
    """
    Computes the heading direction of the group at a given time `t`
    """

    # compute the heading direction of the group
    d_group_t = vel_t.mean(axis=0)

    # normalize to unit vector
    d_group_t = d_group_t / jnp.linalg.norm(d_group_t)
    
    return d_group_t

def compute_group_centroid(pos_t):
    """
    Computes the centroid of the group at a given time `t`
    """

    # compute the centroid of the group
    c_group_t = pos_t.mean(axis=0)

    return c_group_t

def compute_angular_distances(pos_t, d_group_t, c_group_t):
    """
    Computes the angular distances between every individual and the heading-direction weighted center of the group
    """

    # compute the angular distances between every individual and the heading-direction weighted center of the group
    angular_distances_t = jnp.einsum('i,ji->j', d_group_t, pos_t - c_group_t[None,...])

    return angular_distances_t

def compute_rankings(angular_distances_t):
    """
    Computes the rankings of the agents based on their angular distances from the heading-direction weighted center of the group
    """

    # compute the rankings of the agents based on their angular distances from the heading-direction weighted center of the group
    rankings_t = jnp.argsort(angular_distances_t) # note that this will be sorted from lowest angular distance (back of the group) to highest angular distance (front of the group)

    return rankings_t

def compute_Dgroup_and_rankings_t(pos_t, vel_t):
    """
    This function computes the heading direction of the entire group (`d_group`), the centroid (`c_group`), the
    angular distances between every individual and the heading-direction weighted center of the group. 
    """

    # compute the heading direction of the group
    d_group = compute_group_heading_direction(vel_t)

    # compute the centroid of the group
    c_group = compute_group_centroid(pos_t)

    # compute the angular distances between every individual and the heading-direction weighted center of the group
    angular_distances = compute_angular_distances(pos_t, d_group, c_group)

    # compute the rankings of the agents based on their angular distances
    rankings = compute_rankings(angular_distances)

    return d_group, c_group, angular_distances, rankings

def compute_Dgroup_and_rankings_vmapped(pos_hist, vel_hist):
    """
    This function computes the heading direction of the entire group (`d_group`), the centroid (`c_group`), the
    angular distances between every individual and the heading-direction weighted center of the group. =
    It vmaps the function `compute_Dgroup_and_rankings_t` over the time dimension.
    
    Parameters
    ----------
    pos_hist : (T, N, D) array
        History of D-dimensional positions of all N agents over time
    vel_hist : (T, N, D)  array
        History of D-dimensional velocities of all N agents over time
    
    Returns
    -------
    d_group : (T, D) array
        Heading direction of the group at each time step
    c_group : (T, D) array
        Centroid of the group at each time step
    angular_distances : (T, N) array
        Angular distances between every individual and the heading-direction weighted center of the group at each time step
    rankings : (T, N) array
        Rankings of the agents based on the magnitude of their angular distances from the heading-direction weighted center of the group at each time step
    """

    return vmap(compute_Dgroup_and_rankings_t, in_axes=(0, 0), out_axes=(0,0,0,0))(pos_hist, vel_hist)

def compute_integrated_change_magnitude(d_group_reference, d_group_hist):
    """
    This function computes the magnitude of the integrated change in heading direction of the group (stored in rows of `d_group_hist`)
    relative to some reference group heading direction (`d_group_reference`).

    Parameters
    ----------
    d_group_reference : (D,) array
        Reference group heading direction
    d_group_hist : (T, D) array
        History of group heading directions over time
    
    Returns
    -------
    integrated_change_magnitude : float
        Magnitude of the time-integrated change in heading direction of the group over time
    """

    # compute the sum of the dot products / cosine angles over time
    integrated_change_magnitude = -jnp.clip(jnp.dot(d_group_hist, d_group_reference), -1.0, 1.0).sum()

    return integrated_change_magnitude


    



        

    