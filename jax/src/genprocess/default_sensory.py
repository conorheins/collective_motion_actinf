from typing import Any, Callable, Dict, Iterable, Tuple

import jax.numpy as jnp
from . import geometry as geo

array = jnp.ndarray

def identity_transform(x):
    """
    Default transform from hidden states to observations
    """

    return x

def grad_identity(x):
    """
    Gradient of the default identity transform with respect to x
    """

    return 1.

def sensory_samples_multi_order(x_all: Iterable, noise_all: Iterable, g: Callable, dgdx: Callable, remove_zeros: bool = True) -> Tuple[array, array]:
    """
    Function that samples a set of sensory observations from hidden states, 
    given potentially multiple orders of motion in the hidden states
    Arguments
    ========
    `x_all` [container of jnp.ndarrays of same shape]: 
    `noise_all` [container of jnp.ndarrays of same shape]
    """

    x0, noise0 = x_all[0], noise_all[0]
    phi_all_orders = [g(x0) + noise0] + [dgdx(x0) * x_p + noise_p for (x_p, noise_p) in zip(x_all[1:], noise_all[1:])]
    phi_all_orders = jnp.vstack(phi_all_orders)

    empty_sector_mask = jnp.vstack(x_all) == 0.

    phi_final = (phi_all_orders * jnp.logical_not(empty_sector_mask)) if remove_zeros else phi_all_orders

    return phi_final, empty_sector_mask

def get_observations(pos: array, vel: array, genproc: Dict, t_idx: int) -> Tuple[array, array, array]:
    """ 
    Takes arrays of current positions and velocities of all agents and a generative process Dict and a time index
    and returns observations, gradient vectors (used later on for active inference computations) and a mask of empty sectors 
    """

    # compute visual neighbourhoods 
    within_sector_idx, distance_matrix, n2n_vecs = geo.compute_visual_neighbours(pos, vel, genproc['R_starts'], genproc['R_ends'], genproc['dist_thr'])

    # get h (first order observations)
    h = geo.compute_h_per_sector(within_sector_idx, distance_matrix)

    # get hprime (velocity of observations)
    hprime, all_dh_dr_self = geo.compute_hprime_per_sector(within_sector_idx, pos, vel, n2n_vecs)

    # aggregate hidden states
    x_tilde_gp = [h, hprime]

    # sample observations
    phi, empty_sectors_mask = sensory_samples_multi_order(x_tilde_gp, genproc['sensory_noise'][t_idx], genproc['sensory_transform'], genproc['grad_sensory_transform'], remove_zeros=True)

    return phi, all_dh_dr_self, empty_sectors_mask

def get_observations_special(pos: array, vel: array, genproc: Dict, t_idx: int) -> Tuple[array, array, array]:
    """ 
    Takes arrays of current positions and velocities of all agents and a generative process Dict and a time index
    and returns observations, gradient vectors (used later on for active inference computations) and a mask of empty sectors 
    """

    # compute visual neighbourhoods 
    within_sector_idx, distance_matrix, n2n_vecs = geo.compute_visual_neighbours(pos, vel, genproc['R_starts'], genproc['R_ends'], genproc['dist_thr'])

    # get h (first order observations)
    h = geo.compute_h_per_sector(within_sector_idx, distance_matrix)

    # get hprime (velocity of observations)
    hprime, all_dh_dr_self = geo.compute_hprime_per_sector_special(within_sector_idx, pos, vel, n2n_vecs)

    # aggregate hidden states
    x_tilde_gp = [h, hprime]

    # sample observations
    phi, empty_sectors_mask = sensory_samples_multi_order(x_tilde_gp, genproc['sensory_noise'][t_idx], genproc['sensory_transform'], genproc['grad_sensory_transform'], remove_zeros=True)

    return phi, all_dh_dr_self, empty_sectors_mask






