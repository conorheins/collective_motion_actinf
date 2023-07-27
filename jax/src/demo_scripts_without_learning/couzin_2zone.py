""" 
Implementation of the 2-zone Couzin model for simulating schooling / flocking behavior in 2-D, based on the paper:
    Couzin, I. D., et al. (2005). Effective leadership and decision-making in animal groups on the move. Nature, 433(7026), 513-516.
"""
import jax.numpy as jnp
from jax import vmap, random, lax
from jax_md import space
from jax_md.space import map_product, distance
from genprocess import compute_nearest_neighbour_vectors, compute_clockwise_proj_single, advance_positions, normalize_array, safe_normalize

from functools import partial

euclidean_dist_fn, _ = space.free()
euclidean_dist_vmapped = map_product(euclidean_dist_fn)


def rotate(target_vec, current_vec, turn_angle=1, dt=0.01):
    """
    Rotate the current vector by the given angle (in degrees) towards the target vector.
    """
    perp = target_vec - current_vec * jnp.inner(target_vec, current_vec)

    scaled = perp * jnp.sqrt(jnp.inner(current_vec, current_vec))/jnp.clip(jnp.sqrt(jnp.inner(perp, perp)),
    a_min=1e-8)

    return jnp.cos(dt * turn_angle) * current_vec + jnp.sin(dt * turn_angle) * scaled

def extract_neighbours(pos, vel, fop_angle=315.0, dist_thr=7.0):

    # get array of vectors pointing from each SPP to its nearest neighbours
    n2n_vecs = compute_nearest_neighbour_vectors(pos)

    # get the matrix of distances between particles (i.e. the pairwise distances)
    dist_matrix = jnp.linalg.norm(n2n_vecs, axis = 2)

    # normalize the n2n vectors to unit-length
    n2n_vecs /= dist_matrix[...,None]

    # compute the cosine of the angle between the velocity vector of each particle and its nearest neighbour
    cos_angles = compute_clockwise_proj_single(-vel, n2n_vecs)
    angles = jnp.arccos(cos_angles) * 180. / jnp.pi # convert cos angles to angles in degrees
    within_fop = (angles > (360. - fop_angle)) & (dist_matrix < dist_thr) # find all neighbours within the focal agent's field of perception (fop)

    return within_fop, n2n_vecs, dist_matrix

def compute_repulsion_forces_single(n2n_vecs, within_fop, within_repulsion):
    
    # compute the repulsion forces as the sum of the nearest neighbour vectors

    # zero out the rows of `n2n_vecs` that correspond to neighbours that are not within the focal agent's field of perception
    repulsion_forces = jnp.nansum(-n2n_vecs * within_repulsion[...,None], axis = 0)

    # normalize the repulsion forces to unit-length
    vel_next = repulsion_forces / jnp.linalg.norm(repulsion_forces)
    # vel_next = safe_normalize(repulsion_forces)

    return vel_next

def compute_social_forces_single(all_vel, n2n_vecs, within_fop, within_repulsion):
        
    # compute the positive social forces as the sum of the attraction and orientation forces
    attraction_forces = jnp.nansum(n2n_vecs * within_fop[...,None], axis = 0)
    orientation_forces = jnp.nansum(all_vel * within_fop[...,None], axis = 0)

    # compute the social forces as the sum of the attraction and orientation forces
    social_forces = attraction_forces + orientation_forces

    # normalize the social forces to unit-length
    vel_next = social_forces / jnp.linalg.norm(social_forces)
    # vel_next = safe_normalize(social_forces)

    return vel_next

def compute_next_heading(pos, vel, n2n_vecs, within_fop, dist_matrix, repulsion_thr=1.0):

    # find all particles that are too close to each other
    within_repulsion = (dist_matrix < repulsion_thr)
    repulsion_flags = jnp.nansum(within_repulsion, axis = 1) > 1.0 # vector of length (N,) that says whether each particle has neighbours within its repulsion zone (> 1.0 is because the focal particle is included in the sum)
    within_repulsion *= (1. - jnp.eye(pos.shape[0])) # zero out the focal particle's row in the `within_repulsion` matrix

    compute_social_forces_partialled = partial(compute_social_forces_single, vel)
    check_fn = lambda flag_i, n2n_i, fop_mask_i, rep_mask_i: lax.cond(flag_i > 0, compute_repulsion_forces_single, compute_social_forces_partialled, n2n_i, fop_mask_i, rep_mask_i)
    vel_next = vmap(check_fn)(repulsion_flags, n2n_vecs, within_fop, within_repulsion)
    # vel_next = vmap(check_fn)(jnp.zeros(dist_matrix.shape[0]), n2n_vecs, within_fop, within_repulsion) # test by turning off repulsive forces
    # vel_next = vmap(check_fn)(jnp.ones(dist_matrix.shape[0]), n2n_vecs, within_fop, within_repulsion) # test by turning off attraction forces

    return jnp.nan_to_num(vel_next)

def make_step_fn(noise_params, particle_params, dt = 0.01, debug=False):
    """
    Function that creates and returns the single timestep function used to simulate a 2-zone Couzin model simulation,
    given the parameter `noise_params` and `particle_params`.
    """
    
    if debug:
        def return_fn(x):
            return x[0], x[1], x[2], x[3], x[4]
    else:
        def return_fn(x):
            return x[0], x[1]

    def single_timestep(pos, vel, t_idx):

        # find neighbours of each particle
        within_fop, n2n_vecs, dist_matrix = extract_neighbours(pos, vel, particle_params['fop_angle'], particle_params['dist_thr'])

        # compute a boolean vector storing whether agent[i] has any neighbours within its any of its zones (repulsion or social)
        # any_interactions = jnp.nansum(dist_matrix * (1. - jnp.eye(dist_matrix.shape[0])), axis =1) > 0
        any_interactions = (dist_matrix < particle_params['dist_thr']).sum(axis=1) > 1 # vector of length (N,) that says whether each particle has neighbours within any of its interaction radii

        # compute the velocity of each particle
        desired_heading = compute_next_heading(pos, vel, n2n_vecs, within_fop, dist_matrix, repulsion_thr=particle_params['repulsion_thr'])

        # desired_heading = vel * (jnp.nansum(within_fop,axis=1))[...,None] + desired_heading * (1. - (jnp.nansum(within_fop,axis=1))[...,None])
        # compute the angular deviation between the current and desired headings for each particle (clip to -1.0 and +1.0)
        turning_rate = jnp.deg2rad(particle_params['angular_turning_rate']) # in radians per second

        # rotate each vector in `vel` by `turning_rate` radians towards the corresponding vector in `desired_heading`
        vel_next = vmap(rotate, in_axes = (0, 0, None, None))(desired_heading, vel, turning_rate, dt)

        theta = jnp.arccos(jnp.clip(jnp.nansum(desired_heading*vel, axis=1), -1.0, 1.0))
        # turning_rate = dt * (particle_params['angular_turning_rate'] / jnp.absolute(theta)) # fraction of incrememt out of the total angular difference

        # vel_next = jnp.cos(turning_rate*jnp.pi/2.)[...,None]*vel + jnp.sin(turning_rate*jnp.pi/2.)[...,None]*desired_heading
        vel_next = vel_next * (jnp.rad2deg(theta) >= (dt*particle_params['angular_threshold']))[...,None] + desired_heading * (jnp.rad2deg(theta) < (dt*particle_params['angular_threshold']))[...,None]
        vel_next = jnp.nan_to_num(vel_next / jnp.linalg.norm(vel_next, axis = 1, keepdims = True))
        vel_next = vel * jnp.logical_not(any_interactions)[...,None] + vel_next * (any_interactions)[...,None]
        
        # use actions to update generative process
        pos_next = advance_positions(pos, vel_next, noise_params['action_noise'][t_idx], speed=particle_params['speed'], dt = dt)

        return return_fn((pos_next, vel_next, desired_heading, within_fop, dist_matrix))
    
    return single_timestep

def simulate_trajectories(init_state, noise_params, particle_params, dt=0.01, n_timesteps=1000, returns='all', debug=False):

    single_timestep_fn = make_step_fn(noise_params, particle_params, dt, debug=debug)

    if returns == 'all':
        returns = ['pos', 'vel', 'desired_heading', 'within_fop', 'dist_matrix'] if debug else ['pos', 'vel']

    idx_to_return = []
    if 'pos' in returns:
        idx_to_return.append(0)
    if 'vel' in returns:
        idx_to_return.append(1)
    
    if debug:
        if 'desired_heading' in returns:
            idx_to_return.append(2)
        if 'within_fop' in returns:
            idx_to_return.append(3)
        if 'dist_matrix' in returns:
            idx_to_return.append(4)

    def step_fn(carry, t):
        pos_past, vel_past = carry
        out = single_timestep_fn(pos_past, vel_past, t)
        pos, vel = out[0], out[1]
        return_states = tuple(out[idx] for idx in idx_to_return)
        return (pos, vel), return_states

    _, history = lax.scan(step_fn, init_state, jnp.arange(n_timesteps))

    return history