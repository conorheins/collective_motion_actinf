from typing import Any, Callable, Dict

from jax import numpy as jnp
from jax import lax, vmap
from functools import partial

array = jnp.ndarray


def run_inference_as_Pi_z(phi: array, mu_init: array, Pi_z: array, empty_sector_mask: array, genmodel: Dict, k_mu = 0.1, num_steps = 1):
    """ Run inference by scanning over the `single_step_GF` step function, which itself is `partial`'d to make it
    have mostly frozen arguments, and then wrapped in a loose syntax to make it compatible with `lax.scan` """

    GF_step_one_arg = partial(single_step_GF_as_Piz, 
                            phi = phi, # fix observations
                            Pi_z = Pi_z, # fix precision matrices
                            empty_sector_mask = empty_sector_mask, # fix empty sector mask
                            genmodel = genmodel,  # fix generative model while doing inference (aka no learning)
                            step_size = k_mu # fix learning rate while doing inference
                        )

    def f_genfilt(carry, t):

        mu_current, _ = carry # don't need the current prediction error for the next generalised filtering step

        mu_next, epsilon_z_next = GF_step_one_arg(mu_current)

        return (mu_next, epsilon_z_next), mu_next
    
    init_state = (mu_init, jnp.zeros_like(mu_init))

    out_final, mu_traj = lax.scan(f_genfilt, init_state, jnp.arange(0, num_steps))

    return out_final, mu_traj

def run_inference(phi: array, mu_init: array, empty_sector_mask: array, genmodel: Dict, k_mu = 0.1, num_steps = 1):
    """ Run inference by scanning over the `single_step_GF` step function, which itself is `partial`'d to make it
    have mostly frozen arguments, and then wrapped in a loose syntax to make it compatible with `lax.scan` """

    GF_step_one_arg = partial(single_step_GF, 
                            phi = phi, # fix observations
                            empty_sector_mask = empty_sector_mask, # fix empty sector mask
                            genmodel = genmodel,  # fix generative model while doing inference (aka no learning)
                            step_size = k_mu # fix learning rate while doing inference
                        )

    def f_genfilt(carry, t):

        mu_current, _ = carry # don't need the current prediction error for the next generalised filtering step

        mu_next, epsilon_z_next = GF_step_one_arg(mu_current)

        return (mu_next, epsilon_z_next), mu_next
    
    init_state = (mu_init, jnp.zeros_like(mu_init))

    out_final, mu_traj = lax.scan(f_genfilt, init_state, jnp.arange(0, num_steps))

    return out_final, mu_traj

def matrix_vec(matrix: array, vector: array):
    return matrix @ vector

def zero_out(arr: array, remove_mask: array):
    """ 
    Given an array and a remove_mask, this function finds all values in the indices of the array 
    encoded by 1 entries in `remove_mask` and replaces them with 0's 
    """

    return arr * jnp.logical_not(remove_mask)

def single_step_GF(mu: array, phi: array, empty_sector_mask: array, genmodel: Dict, step_size: float = 0.1):
    """ Single step of generalised filtering or predictive coding in generalised coordinates of motion, for multiple particles or agents """

    g, grad_g, g_params = genmodel['g'], genmodel['grad_g'], genmodel['g_params']
    g_vm = vmap(g, (1, 0), 1) # assumes input state dimension (mu in this case) is in the columns (1th dimension), and parameters of g are in the 0th dimension
    grad_g_vm = vmap(grad_g, (1, 0), 0)

    Pi_z = genmodel['Pi_z'] # each Pi_z[i] are the sensory precision parameters of a particular agent

    f, grad_f, f_params = genmodel['f'], genmodel['grad_f'], genmodel['f_params']
    f_vm = vmap(f, (1, 0), 1) # assumes input state dimension (mu in this case) is in the columns (1th dimension), and parameters of f are in the 0th dimension
    grad_f_vm = vmap(grad_f, (1, 0), 0)

    Pi_w = genmodel['Pi_w'] # each Pi_w[i] are the process precision parameters of a particular agent

    D_shift, D_T = genmodel['D_shift'], genmodel['D_T']

    """ Step 1. Compute sensory prediction error component of the gradient of Laplace VFE with respect to mu """
    # sensory prediction errors 
    s_pe = phi - g_vm(mu, g_params) # this vmaps g across agent dimension in both `mu` and `g_params["paramX"]`
    s_pe = zero_out(s_pe, empty_sector_mask) # zero out empty sectors

    # precision weight the sensory prediction errors
    p_weighted_spe = vmap(matrix_vec, (0, 1), 1)(Pi_z, s_pe) # dots each agent-specific Pi_z[i] matrix with each agent-specific s_pe[:,i] sensory prediction error
    p_weighted_spe = zero_out(p_weighted_spe, empty_sector_mask) # zero out empty sectors

    # use chain rule to multiply precision-weighted sensory predicion errors by gradient of sensory function
    grad_g_eval = grad_g_vm(mu, g_params)
    epsilon_z = vmap(matrix_vec, (0, 1), 1)(grad_g_eval, p_weighted_spe)

    """ Step 2. Compute "model" or "process" prediction error component of the gradient of Laplace VFE with respect to mu """
    # process prediction errors 
    p_pe = D_shift @ mu - f_vm(mu, f_params) # this vmaps f across agent dimension in both `mu` and `f_params["paramX"]`
    # p_pe = vmap(matrix_vec, (0, 1), 1)(D_shift, mu) - f_vm(mu, f_params)

    # precision weight the process prediction errors
    p_weighted_ppe = vmap(matrix_vec, (0, 1), 1)(Pi_w, p_pe) # dots each agent-specific Pi_w[i] matrix with each agent-specific p_pe[:,i] process prediction error

    # use chain rule to multiply precision-weighted process predicion errors by gradient of dynamics or "flow" function
    grad_f_eval = grad_f_vm(p_weighted_ppe, f_params)
    epsilon_w = vmap(matrix_vec, (0, 1), 1)(grad_f_eval, p_weighted_ppe)  - D_T @ p_weighted_ppe # this vmaps grad_f @ p_weighted_ppe across agent dimension, allows individualized grad_f
    # epsilon_w = vmap(matrix_vec, (0, 1), 1)(grad_f_eval, p_weighted_ppe)  - vmap(matrix_vec, (0, 1), 1)(D_T, p_weighted_ppe) # this vmaps grad_f @ p_weighted_ppe across agent dimension, allows individualized grad_f

    """ Step 3. combine sensory and model prediction errors (with generalised correction term) to compute the total increment to mu """
    dMu_dt = D_shift @ mu + epsilon_z + epsilon_w
    # dMu_dt = vmap(matrix_vec, (0, 1), 1)(D_shift, mu) + epsilon_z + epsilon_w
    mu_new = mu + (step_size * dMu_dt) # update using learning rate given by `step_size`

    return mu_new, epsilon_z


def single_step_GF_as_Piz(mu: array, phi: array, Pi_z: array, empty_sector_mask: array, genmodel: Dict, step_size: float = 0.1):
    """ Single step of generalised filtering or predictive coding in generalised coordinates of motion, for multiple particles or agents """

    g, grad_g, g_params = genmodel['g'], genmodel['grad_g'], genmodel['g_params']
    g_vm = vmap(g, (1, 0), 1) # assumes input state dimension (mu in this case) is in the columns (1th dimension), and parameters of g are in the 0th dimension
    grad_g_vm = vmap(grad_g, (1, 0), 0)

    f, grad_f, f_params = genmodel['f'], genmodel['grad_f'], genmodel['f_params']
    f_vm = vmap(f, (1, 0), 1) # assumes input state dimension (mu in this case) is in the columns (1th dimension), and parameters of f are in the 0th dimension
    grad_f_vm = vmap(grad_f, (1, 0), 0)

    Pi_w = genmodel['Pi_w'] # each Pi_w[i] are the process precision parameters of a particular agent

    D_shift, D_T = genmodel['D_shift'], genmodel['D_T']

    """ Step 1. Compute sensory prediction error component of the gradient of Laplace VFE with respect to mu """
    # sensory prediction errors 
    s_pe = phi - g_vm(mu, g_params) # this vmaps g across agent dimension in both `mu` and `g_params["paramX"]`
    s_pe = zero_out(s_pe, empty_sector_mask) # zero out empty sectors

    # precision weight the sensory prediction errors
    p_weighted_spe = vmap(matrix_vec, (0, 1), 1)(Pi_z, s_pe) # dots each agent-specific Pi_z[i] matrix with each agent-specific s_pe[:,i] sensory prediction error
    p_weighted_spe = zero_out(p_weighted_spe, empty_sector_mask) # zero out empty sectors

    # use chain rule to multiply precision-weighted sensory predicion errors by gradient of sensory function
    grad_g_eval = grad_g_vm(mu, g_params)
    epsilon_z = vmap(matrix_vec, (0, 1), 1)(grad_g_eval, p_weighted_spe)

    """ Step 2. Compute "model" or "process" prediction error component of the gradient of Laplace VFE with respect to mu """
    # process prediction errors 
    p_pe = D_shift @ mu - f_vm(mu, f_params) # this vmaps f across agent dimension in both `mu` and `f_params["paramX"]`
    # p_pe = vmap(matrix_vec, (0, 1), 1)(D_shift, mu) - f_vm(mu, f_params)

    # precision weight the process prediction errors
    p_weighted_ppe = vmap(matrix_vec, (0, 1), 1)(Pi_w, p_pe) # dots each agent-specific Pi_w[i] matrix with each agent-specific p_pe[:,i] process prediction error

    # use chain rule to multiply precision-weighted process predicion errors by gradient of dynamics or "flow" function
    grad_f_eval = grad_f_vm(p_weighted_ppe, f_params)
    epsilon_w = vmap(matrix_vec, (0, 1), 1)(grad_f_eval, p_weighted_ppe)  - D_T @ p_weighted_ppe # this vmaps grad_f @ p_weighted_ppe across agent dimension, allows individualized grad_f
    # epsilon_w = vmap(matrix_vec, (0, 1), 1)(grad_f_eval, p_weighted_ppe)  - vmap(matrix_vec, (0, 1), 1)(D_T, p_weighted_ppe) # this vmaps grad_f @ p_weighted_ppe across agent dimension, allows individualized grad_f

    """ Step 3. combine sensory and model prediction errors (with generalised correction term) to compute the total increment to mu """
    dMu_dt = D_shift @ mu + epsilon_z + epsilon_w
    # dMu_dt = vmap(matrix_vec, (0, 1), 1)(D_shift, mu) + epsilon_z + epsilon_w
    mu_new = mu + (step_size * dMu_dt) # update using learning rate given by `step_size`

    return mu_new, epsilon_z






















