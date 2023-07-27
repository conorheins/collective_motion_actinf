from jax import numpy as jnp
from jax.scipy.linalg import block_diag

def f_base_linear(x0, params):
    """ Basic linear flow function for lowest-order states """
    return -params['A0'] @ (x0 - params['eta0'])

def f_tilde_gen(tilde_x, f0, params):
    """ Generalised flow function with arbitrary functional form """
    pass

def f_tilde_linear(tilde_x, params):
    """ 
    Generalised linear flow function
    `tilde_x` [jnp.ndarray of size (n_x * n_do_x, )]: 
    `params`: [dict with two keys]:
                `tilde_A`  : [jnp.ndarray of size (n_do, n_x, n_x)]
                `tilde_eta`: [jnp.ndarray of size(n_do, n_x)]
    """

    generalised_flow = -block_diag(*params['tilde_A']) # this assumes no interactions between orders in the flow

    return generalised_flow @ (tilde_x - params['tilde_eta'].flatten())

def grad_f_tilde_linear(tilde_x, params):

    grad_f = (-block_diag(*params['tilde_A'])).T # this assumes no interactions between orders in the flow

    return grad_f

def g_base_linear(x0, params):
    """ Basic linear sensory function for lowest-order states """
    return -params['g0'] @ x0

def g_tilde_gen(tilde_x, g0, params):
    """ Generalised sensory function with arbitrary functional form """
    pass

def g_tilde_linear(tilde_x, params):
    """
    Generalised linear sensory function
    `tilde_x` [jnp.ndarray of size (n_x * n_do_x, )]: 
    `params`: [dict with one key]:
                `tilde_g`  : [jnp.ndarray of size (n_do_phi, n_phi, n_phi)]
    """
    generalised_g = block_diag(*params['tilde_g'])
    remaining_columns = jnp.zeros((generalised_g.shape[0], tilde_x.shape[0] - generalised_g.shape[1]))
    generalised_g = jnp.hstack( (generalised_g, remaining_columns) )

    return generalised_g @ tilde_x

def grad_g_tilde_linear(tilde_x, params):

    generalised_g_T = block_diag(*params['tilde_g']).T
    remaining_rows = jnp.zeros((tilde_x.shape[0] - generalised_g_T.shape[0], generalised_g_T.shape[1]))
    grad_g = jnp.vstack( (generalised_g_T, remaining_rows) )

    return grad_g

def parameterize_A0_no_coupling(alpha, ns_x):
    """ 
    Function that takes a scalar value alpha and uses it to create a (ns_x, ns_x)-shaped flow matrix
    """
    A0 = alpha * jnp.eye(ns_x)
    return A0

def parameterize_A0_with_coupling(alpha_beta, ns_x):
    """ 
    Function that takes a size-(2,) vector of an alpha and a beta, and uses them to create (ns_x, ns_x)-shaped flow matrix
    """
    alpha, beta = alpha_beta[0], alpha_beta[1]
    A0 = alpha * jnp.eye(ns_x) + beta * jnp.eye(ns_x, k = 1) + beta * jnp.eye(ns_x, k = -1)
    return A0







    
    

