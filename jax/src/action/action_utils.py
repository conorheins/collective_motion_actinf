from typing import Any, Callable, Dict

from jax import numpy as jnp
from jax import lax, vmap
from functools import partial

array = jnp.ndarray

def remove_nans(arr):
    """ Replaces NaN-valued elements of an array with 0.0's """
    return jnp.nan_to_num(arr)

def normalize_array(array, axis = 1):
    """
    Normalize an array along some given axis so that it has unit normed vectors stored along that dimension
    """

    norms = jnp.sqrt((array**2).sum(axis = axis))
    reshape_dims = [(1 if ii == axis else dim) for ii, dim in enumerate(array.shape)]
    return jnp.divide(array, norms.reshape(reshape_dims))


def infer_actions(v: array, epsilon_z: array, genmodel: Dict, all_dh_dr_self: array, k_alpha = 0.1, num_steps = 1, normalize_v = True):
    """ Run inference by scanning over the `single_step_GF` step function, which itself is `partial`'d to make it
    have mostly frozen arguments, and then wrapped in a loose syntax to make it compatible with `lax.scan` """

    ns_phi, ndo_phi = genmodel['ns_phi'], genmodel['ndo_phi']
    epsilon_z_prime = epsilon_z[ns_phi:(ns_phi*ndo_phi)] # gradient of VFE w.r.t to velocity observations, shape = (n_sectors, N)

    action_step_one_arg = partial(update_action_identity_g, 
                            dF_dPhiprime = epsilon_z_prime, # fix prediction error
                            dPhiprime_dv = all_dh_dr_self, # fix sector vectors
                            genmodel = genmodel,  # fix generative model while inferring action (aka no learning)
                            step_size = k_alpha # fix learning rate while doing inference
                        )

    def f_actionupdate(carry, t):

        v_current = carry # don't need the current prediction error for the next generalised filtering step

        v_next = action_step_one_arg(v_current)

        return v_next, v_next
    
    v_final, _ = lax.scan(f_actionupdate, v, jnp.arange(0, num_steps))

    return normalize_array(v_final, axis = 1) if normalize_v else v_final

def update_action_identity_g(v: array, dF_dPhiprime: array, dPhiprime_dv: array, genmodel: Dict, step_size = 0.1):
    """
    Vectorized implementation of computing the gradients of free energy with respect to actions, computed across individuals.
    Assumptions: 1) g(x0) = x0 ==> dg(x0)/dx0 = 1.0; 
                 2) g(x_prime) = dg(x0)/dx0 x_prime 
    This implies that dg(x_prime)/dv =  d(dg(x0)/dx0 x_prime) / dv = dg(x0)/dx0 * dh_dr_self = dh_dr_self, because dg(x0)/dx0 = 1.0; 
    where dh_dr_self is the set of vectors pointing towards the average neighbour wthin each sector (arrive at this by differentiating x_prime with respect to v)
    """

    # dF_dv = (dF_dPhiprime[...,None] * remove_nans(dPhiprime_dv)).sum(axis=0) # shape should be (N, 2). @NOTE: Need to use this one to be able to work with auto-differentiation. This is a vectorized linear combination of each sector vector, using the sensory prediction error for that sector as combination weights
    dF_dv = remove_nans(dF_dPhiprime[...,None] * dPhiprime_dv).sum(axis=0) # shape should be (N, 2). This is a vectorized linear combination of each sector vector, using the sensory prediction error for that sector as combination weights

    v_new = v - (step_size * dF_dv) # update using learning rate given by `step_size`

    return v_new
