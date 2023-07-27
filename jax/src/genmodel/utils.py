from jax import numpy as jnp
from jax import vmap

from .defaults import f_tilde_linear, grad_f_tilde_linear, g_tilde_linear, grad_g_tilde_linear, parameterize_A0_no_coupling
from .precisions import create_full_precision_matrix

def init_genmodel(initialization_dict):
    """ Function for initializing the generative model """

    # extract variables that will be useful
    N, ns_x, ns_phi, ndo_x, ndo_phi = initialization_dict['N'], initialization_dict['ns_x'], initialization_dict['ns_phi'], initialization_dict['ndo_x'], initialization_dict['ndo_phi']
    alpha, eta, pi_z_spatial, pi_w_spatial = initialization_dict['alpha'], initialization_dict['eta'], initialization_dict['pi_z_spatial'], initialization_dict['pi_w_spatial']
    s_z, s_w = initialization_dict['s_z'], initialization_dict['s_w']
    # initialize genmodel dict
    genmodel = {}

    # initialize flow functions and parameters thereof
    genmodel['f'] = f_tilde_linear
    genmodel['grad_f'] = grad_f_tilde_linear
    genmodel['f_params'] = {}

    A0_all = vmap(parameterize_A0_no_coupling, (0, None))(alpha * jnp.ones(N), ns_x)
    genmodel['f_params']['tilde_A'] = jnp.stack(ndo_x * [A0_all], axis = 1)

    eta0_all = eta * jnp.ones((N,1,ns_x))
    genmodel['f_params']['tilde_eta'] = jnp.concatenate((eta0_all, jnp.zeros((N, ndo_x-1, ns_x))), axis = 1)
    genmodel['Pi_w'] = vmap(create_full_precision_matrix, (None, None, 0, 0))(ns_x, ndo_x, pi_w_spatial * jnp.ones(N), s_w * jnp.ones(N))

    #initialize shift matrices
    genmodel['D_shift'] = jnp.diag(jnp.ones((ns_x*ndo_x- ns_x)), k = ns_x)
    genmodel['D_T'] = genmodel['D_shift'].T

    # initialize sensory likelihood functions and parameters thereof
    genmodel['g'] = g_tilde_linear
    genmodel['grad_g'] = grad_g_tilde_linear
    genmodel['g_params'] = {}

    g0_all = jnp.stack(N * [jnp.eye(ns_phi)], axis = 0)
    genmodel['g_params']['tilde_g'] = jnp.stack(ndo_phi * [g0_all], axis = 1)
    genmodel['Pi_z'] = vmap(create_full_precision_matrix, (None, None, 0, 0))(ns_phi, ndo_phi, pi_z_spatial * jnp.ones(N), s_z * jnp.ones(N))

    # put dimension variables into generative model
    genmodel['ns_phi'], genmodel['ndo_phi'], genmodel['ns_x'], genmodel['ndo_x'] = ns_phi, ndo_phi, ns_x, ndo_x

    return genmodel