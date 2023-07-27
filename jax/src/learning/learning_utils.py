from genmodel import compute_vfe_single, compute_vfe_single_all_params
from genmodel.precisions import create_temporal_precisions

from funcy import project, merge
from functools import partial

from jax import grad, vmap, jit, lax, tree_util
from jax import numpy as jnp

def parameterize_pi_z(s_z, pi_z_spatial, ns_phi, ndo_phi):
    spatial = pi_z_spatial * jnp.eye(ns_phi)
    # temporal = create_temporal_precisions(ndo_phi, s_z)[0] # currently this isn't working so replaced it with an analytic expression for the smoothness that only works when ndo_phi = 2
    temporal = jnp.diag(jnp.array([1.0, 2 * s_z**2]))
    return jnp.kron(temporal, spatial)

def parameterize_pi_w(s_w, pi_w_spatial, ns_x, ndo_x):
    spatial = pi_w_spatial * jnp.eye(ns_x)
    # temporal = create_temporal_precisions(ndo_x, s_w)[0] # currently this isn't working so replaced it with an analytic expression for the smoothness that only works when ndo_x = 3
    temporal = jnp.diag(jnp.array([1.5, 2 * s_w**2, 2*s_w**4])) + jnp.diag(jnp.array([s_w**2]), k = 2) + jnp.diag(jnp.array([s_w**2]), k = -2)
    return jnp.kron(temporal, spatial)

def update_sz(s_z, mu, phi, empty_sectors_mask, genmodel, k_param = 0.01, num_steps = 1):
    
    # get some constants
    D_shift, g, f = genmodel['D_shift'], genmodel['g'], genmodel['f']
    pi_z_spatial, ns_phi, ndo_phi = genmodel['pi_z_spatial'], genmodel['ns_phi'], genmodel['ndo_phi']

    def vfe_function(s_z, Pi_w, g_params, f_params, mu, phi, empty_sectors_mask):
        Pi_z = parameterize_pi_z(s_z, pi_z_spatial, ns_phi, ndo_phi)
        return compute_vfe_single_all_params(mu, phi, empty_sectors_mask, D_shift = D_shift, Pi_z = Pi_z, Pi_w = Pi_w, g_params = g_params, f_params = f_params, g = g, f = f)
                           
    grad_fn = grad(vfe_function, argnums=0) # get gradient of vfe function of smoothness with respect to smoothness argument

    grad_fn_vmapped_and_jitted = jit(vmap(grad_fn, (0, 0, 0, 0, 1, 1, 1), 0))
    # dF_dsz = grad_fn_vmapped_and_jitted(s_z, genmodel['Pi_w'], genmodel['g_params'], genmodel['f_params'], mu, phi, empty_sectors_mask)
    # s_z -= k_param * dF_dsz # vectorized update to everyone's smoothness parameter

    def s_z_udpate(carry, t):
        s_z_last = carry
        s_z_next = s_z_last - k_param * grad_fn_vmapped_and_jitted(s_z_last, genmodel['Pi_w'], genmodel['g_params'], genmodel['f_params'], mu, phi, empty_sectors_mask)
        return s_z_next, s_z_next
    
    out, _ = lax.scan(s_z_udpate, s_z, jnp.arange(num_steps))

    return out

def update_sw(s_w, mu, phi, empty_sectors_mask, genmodel, k_param = 0.01, num_steps = 1):
    
    # get some constants
    D_shift, g, f = genmodel['D_shift'], genmodel['g'], genmodel['f']
    pi_w_spatial, ns_x, ndo_x = genmodel['pi_z_spatial'], genmodel['ns_phi'], genmodel['ndo_phi']

    def vfe_function(s_w, Pi_z, g_params, f_params, mu, phi, empty_sectors_mask):
        Pi_w = parameterize_pi_w(s_z, pi_w_spatial, ns_x, ndo_x)
        return compute_vfe_single_all_params(mu, phi, empty_sectors_mask, D_shift = D_shift, Pi_z = Pi_z, Pi_w = Pi_w, g_params = g_params, f_params = f_params, g = g, f = f)
                           
    grad_fn = grad(vfe_function, argnums=0) # get gradient of vfe function of smoothness with respect to smoothness argument

    dF_dsw = vmap(grad_fn, (0, 0, 0, 0, 1, 1, 1), 0)(s_w, genmodel['Pi_z'], genmodel['g_params'], genmodel['f_params'], mu, phi, empty_sectors_mask)

    s_w -= k_param * dF_dsw # vectorized update to everyone's smoothness parameter

    return s_w

# def parameterize_args(params, arg_param_funcs):
#     """ Parameterize all the parameters in `params` into a dictionary of arguments to be passed to `compute_vfe_single_all_params` """

#     for param_name, value in learnable_params.items():
        
#         arg_name, parameterize = arg_param_funcs[param_name]['arg_name'], arg_param_funcs[param_name]['fn']
#         out_dict[arg_name] = parameterize(value)
    
#     return out_dict


# def update_parameters(mu, phi, empty_sectors_mask, parameters_to_update, arg_param_funcs, learnable_params, static_args, k_params = 0.01, num_steps = 1):
#     """ Attempt at a generic parameter update function """

#     def vfe_fn(learnable_params):

#         learnable_args = parameterize_args(learnable_params, arg_param_funcs)

#         return compute_vfe_single_all_params(mu, phi, empty_sectors_mask, **learnable_args, **static_args)   

#     dF_dparams_fn = vmap(grad(vfe_function)) # vmap goes across individuals

#     subset_params = project(learnable_params, parameters_to_update)
#     subset_gradients = project(dF_dparams_fn(learnable_params), parameters_to_update)
#     subset_updated_params = tree_util.tree_multimap(lambda p, g: p - k_params * g, subset_params, subset_gradients)

#     return merge(learnable_params, subset_updated_params)

    ### SCAN VERSION: TEST LATER
    # def update(params, gradients):

        # subset_params = project(learnable_params, parameters_to_update) # subset the dict by the keys corresponding to parameters you want to learn
        # subset_gradients = project(gradients, parameters_to_update)     # subset the gradients by the keys corresponding to parameters you want to learn
        # return tree_util.tree_multimap(lambda param, gradient: param - k_params * gradient, subset_params, subset_gradients)


    # def fn_to_scan(carry, t):

    #     params_last = carry
    #     dF_dparams_eval = dF_dparams_fn(params_last)
    #     params_updated_subset = update(params_last, dF_dparams_eval)
    #     params_next = merge(learnable_params, params_updated_subset)

    #     return params_next, params_next

    # out, _ = lax.scan(fn_to_scan, learnable_params, jnp.arange(num_steps))

    # return out

def find_first_true_index(bool_list):

    i = 0
    match_not_found = True
    while match_not_found:

        if bool_list[i]:
            match_not_found = False
            return i
        i += 1

    return i

def get_batch_dimension(leaf, N):

    if isinstance(leaf, jnp.ndarray):

        leaf_matches = [shape_i == N for shape_i in leaf.shape]

        try:
            batch_axis = find_first_true_index(leaf_matches)
        except:
            batch_axis = None

        return batch_axis

    else:
        return None

def get_vmap_axes(some_pytree, N):
    return tree_util.tree_map(partial(get_batch_dimension, N = N), some_pytree)

def complement(some_dict, key_names):

    out = []
    for key in some_dict.keys():
        if key not in key_names:
            out.append(key)

    return out

def split_params(full_dict, learnable_key_names):

    learnable_params = project(full_dict, learnable_key_names)

    fixed_key_names = complement(full_dict, learnable_key_names)
    fixed_params = project(full_dict, fixed_key_names)

    return learnable_params, fixed_params


def reparameterize(params_to_learn, parameterization_mapping):
    """ Parameterize all the parameters in `params_to_learn` into a dictionary of arguments to be passed to `local_loss_fn`.
    Uses keys of `parameterization_mapping` (whose keys are equal to names in `params_to_learn`) to get the out-arg name in 
    the `reparamd_dict` (which is ultimately an argument to `local_loss_fn`) and a reparameterization function that's used
    to transform the value in params_to_learn into its appropriate form for evaluation in `local_loss_fn`"""

    reparam_dict = {}

    for param_name, value in params_to_learn.items():
        
        to_arg_name, parameterize_fn = parameterization_mapping[param_name]['to_arg_name'], parameterization_mapping[param_name]['fn']
        reparam_dict[to_arg_name] = parameterize_fn(value)
    
    return reparam_dict

def update_parameters(obs, mu, empty_sectors_mask, parameters_to_learn, dFdparam_function, num_steps = 1, k_params = 0.01):

    def param_update_step(carry, t):

        last_params = carry
        dFdparams = dFdparam_function(last_params, obs, mu, empty_sectors_mask)
        updated_params = tree_util.tree_map(lambda param, gradient: param - k_params * gradient, last_params, dFdparams)
        return updated_params, updated_params

    out, _ = lax.scan(param_update_step, parameters_to_learn, jnp.arange(num_steps))

    return out
    
def make_dfdparams_fn(genmodel_dict, preparams, parameterization_mapping, N):
    """
    Creates a function that computes the (vmapped, batched) derivative of vfe with respect to agent-wise generative model parameters,
    given by `preparams`
    """

    learnable_key_names = [parameterization_mapping[preparam_name]['to_arg_name'] for preparam_name in preparams.keys()]
    _, fixed_params = split_params(genmodel_dict, learnable_key_names)

    def vfe_fn_params(frozen_params, params_to_learn, obs, mu, empty_sectors_mask):
        reparameterized_params_to_learn = reparameterize(params_to_learn, parameterization_mapping)
        genmodel = merge(frozen_params, reparameterized_params_to_learn)
        return compute_vfe_single(obs, mu, empty_sectors_mask, genmodel)
    
    vmap_axes_learnable = get_vmap_axes(preparams, N)
    vmap_axes_fixed = get_vmap_axes(fixed_params, N)

    dFdparams_fn = vmap(grad(vfe_fn_params, argnums = 1), (vmap_axes_fixed, vmap_axes_learnable, 1, 1, 1))

    return partial(dFdparams_fn, fixed_params) # compare this to the jitted version, should not matter though




        







    

    
