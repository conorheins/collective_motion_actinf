from jax import numpy as jnp
from jax import vmap
from jax.numpy.linalg import slogdet
from typing import Any, Callable, Dict

array = jnp.ndarray

def log_det(arr: array):
    """ return just the log determinant part of the jnp.linalg.slogdet function's output """
    return slogdet(arr)[1] 

def matrix_vec(matrix: array, vector: array):
    return matrix @ vector

def zero_out(arr: array, remove_mask: array):
    """ 
    Given an array and a remove_mask, this function finds all values in the indices of the array 
    encoded by 1 entries in `remove_mask` and replaces them with 0's 
    """

    return arr * jnp.logical_not(remove_mask)

def compute_vfe_single_all_params(mu: array, phi: array, empty_sectors_mask: array, D_shift: array,
                    Pi_z: array, Pi_w: array, g: Callable, f: Callable, g_params: Dict, f_params: Dict):
    
    """ Step 1. Compute sensory prediction error component of the Laplace VFE """
    s_pe = phi - g(mu, g_params) # sensory prediction errors (ascending observation minus descending prediction)
    s_pe = zero_out(s_pe, empty_sectors_mask) # zero out empty sectors

    # precision weight the sensory prediction errors
    p_weighted_spe = Pi_z @ s_pe # dots a single Pi_z matrix with a single sensory prediction error s_pe 
    p_weighted_spe = zero_out(p_weighted_spe, empty_sectors_mask) # zero out empty sectors

    squared_spes = s_pe.dot(p_weighted_spe) # sensory prediction error component of the variational free energy

    """ Step 2. Compute "model" or "process" prediction error component of the gradient of Laplace VFE with respect to mu """
    # process prediction errors 
    p_pe = D_shift @ mu - f(mu, f_params) #  process prediction errors

    # precision weight the process prediction errors
    p_weighted_ppe = Pi_w @ p_pe # dotsa single Pi_w matrix with a single process prediction error p_pe

    squared_ppes = p_pe.dot(p_weighted_ppe) # process prediction error component of the variational free energy

    """ Step 3. Compute the variance term that is a sum of the log determinants of the variances """

    variance_term = -log_det(Pi_z)-log_det(Pi_w)

    """ Step 4. combine all terms to compute the total variational free energy"""

    vfe = 0.5 * (squared_spes + squared_ppes + variance_term)

    return vfe

def compute_vfe_single(phi: array, mu: array, empty_sectors_mask: array, genmodel: Dict):
    """ 
    Computes Laplace VFE for a single agent 
    (assumes mu and phi are vectors, genmodel['X_params'] are also single matrices, NOT batches of matrices)
    """

    g, g_params = genmodel['g'], genmodel['g_params']
    Pi_z = genmodel['Pi_z'] # each Pi_z[i] are the sensory precision parameters of a particular agent

    f, f_params = genmodel['f'], genmodel['f_params']
    Pi_w = genmodel['Pi_w'] # each Pi_w[i] are the process precision parameters of a particular agent

    D_shift = genmodel['D_shift']

    """ Step 1. Compute sensory prediction error component of the Laplace VFE """
    s_pe = phi - g(mu, g_params) # sensory prediction errors (ascending observation minus descending prediction)
    s_pe = zero_out(s_pe, empty_sectors_mask) # zero out empty sectors

    # precision weight the sensory prediction errors
    p_weighted_spe = Pi_z @ s_pe # dots a single Pi_z matrix with a single sensory prediction error s_pe 
    p_weighted_spe = zero_out(p_weighted_spe, empty_sectors_mask) # zero out empty sectors

    squared_spes = s_pe.dot(p_weighted_spe) # sensory prediction error component of the variational free energy

    """ Step 2. Compute "model" or "process" prediction error component of the gradient of Laplace VFE with respect to mu """
    # process prediction errors 
    p_pe = D_shift @ mu - f(mu, f_params) #  process prediction errors

    # precision weight the process prediction errors
    p_weighted_ppe = Pi_w @ p_pe # dotsa single Pi_w matrix with a single process prediction error p_pe

    squared_ppes = p_pe.dot(p_weighted_ppe) # process prediction error component of the variational free energy

    """ Step 3. Compute the variance term that is a sum of the log determinants of the variances """

    variance_term = -log_det(Pi_z)-log_det(Pi_w)

    """ Step 4. combine all terms to compute the total variational free energy"""

    vfe = 0.5 * (squared_spes + squared_ppes + variance_term)

    return vfe


def compute_vfe_vectorized(mu: array, phi: array, empty_sectors_mask: array, genmodel: Dict):
    """ 
    Computes Laplace VFE vectorized across a batch of agents
    (assumes mu and phi are arrays with agent dimension in the columns, genmodel['X_params'] are batches of tensors with agent dimension in the rows (first dimension)
    """
    g, g_params = genmodel['g'], genmodel['g_params']
    g_vm = vmap(g, (1, 0), 1) # assumes input state dimension (mu in this case) is in the columns (1th dimension), and parameters of g are in the 0th dimension

    Pi_z = genmodel['Pi_z'] # each Pi_z[i] are the sensory precision parameters of a particular agent

    f, f_params = genmodel['f'], genmodel['f_params']
    f_vm = vmap(f, (1, 0), 1) # assumes input state dimension (mu in this case) is in the columns (1th dimension), and parameters of f are in the 0th dimension

    Pi_w = genmodel['Pi_w'] # each Pi_w[i] are the process precision parameters of a particular agent

    D_shift = genmodel['D_shift']

    """ Step 1. Compute sensory prediction error component of the Laplace VFE """
    s_pe = phi - g_vm(mu, g_params) # this vmaps g across agent dimension in both `mu` and `g_params["paramX"]`
    s_pe = zero_out(s_pe, empty_sectors_mask) # zero out empty sectors

    # precision weight the sensory prediction errors
    p_weighted_spe = vmap(matrix_vec, (0, 1), 1)(Pi_z, s_pe) # dots each agent-specific Pi_z[i] matrix with each agent-specific s_pe[:,i] sensory prediction error
    p_weighted_spe = zero_out(p_weighted_spe, empty_sectors_mask) # zero out empty sectors

    squared_spes = (s_pe * p_weighted_spe).sum(axis=0) # vectorized sensory prediction error component of the variational free energy

    """ Step 2. Compute "model" or "process" prediction error component of the gradient of Laplace VFE with respect to mu """
    # process prediction errors 
    p_pe = D_shift @ mu - f_vm(mu, f_params) # this vmaps f across agent dimension in both `mu` and `f_params["paramX"]`

    # precision weight the process prediction errors
    p_weighted_ppe = vmap(matrix_vec, (0, 1), 1)(Pi_w, p_pe) # dots each agent-specific Pi_w[i] matrix with each agent-specific p_pe[:,i] process prediction error

    squared_ppes = (p_pe * p_weighted_ppe).sum(axis=0) # vectorized sensory prediction error component of the variational free energy

    """ Step 3. Compute the variance term that is a sum of the log determinants of the variances """

    logdet_vm = vmap(log_det)
    variance_term = -logdet_vm(Pi_z)-logdet_vm(Pi_w)

    """ Step 4. combine all terms to compute the total variational free energy"""

    vfe = 0.5 * (squared_spes + squared_ppes + variance_term)
    # vfe = 0.5 * (squared_spes + squared_ppes)


    return vfe
    