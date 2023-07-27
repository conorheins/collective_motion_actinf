"""
Experimental script that we can use to automatically identify state variables 
that would lead to the largest change in group free energy, by taking advantage of JAX's modern 
autodifferentiation tools
"""

from jax import random, lax, vmap, jit, grad, jacfwd
from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


from utils import make_single_timestep_fn_nolearning, initialize_meta_params, get_default_inits
from genprocess import init_gen_process
from genmodel import init_genmodel

# set up some global stuff (random key, T, dt, N, D)
key = random.PRNGKey(1)
T, dt, N, D = 50, 0.01, 100, 2
init_dict = get_default_inits(N, T, dt)

# initialize generative model, generative process, and meta parameters related to learning and inference
pos, vel, genproc, new_key = init_gen_process(key, init_dict)
genmodel = init_genmodel(init_dict)
meta_params = initialize_meta_params(infer_lr = 0.1, 
                                    nsteps_infer = 1, 
                                    action_lr = 0.1, 
                                    nsteps_action = 1, 
                                    normalize_v = True
                                    )

# initialize first beliefs using priors over hidden states
mu_init = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

# get single timestep function (no learning version)
single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)

# create a custom step function that will run the simulation as you want
def step_fn(carry, t):
    pos_past, vel_past, mu_past = carry
    pos, vel, mu, F = single_timestep(pos_past, vel_past, mu_past, t)
    return (pos, vel, mu), (pos, vel, mu, F)
init_state = (pos, vel, mu_init)
final_state, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))
pos_hist, vel_hist, mu_hist = history[0], history[1], history[2]

t = 1000

# fig, axes = plt.subplots(1, 2, figsize=(10,8))
# axes[0].quiver(pos_hist[t,:,0], pos_hist[t,:,1], vel_hist[t,:,0], vel_hist[t,:,1])
# angles = jnp.arctan2(vel_hist[t,:,1], vel_hist[t,:,0])
# vel_reconstructed = jnp.concatenate([jnp.cos(angles).reshape(-1,1), jnp.sin(angles).reshape(-1,1)], axis = 1)
# axes[1].quiver(pos_hist[t,:,0], pos_hist[t,:,1], vel_reconstructed[:,0], vel_reconstructed[:,1])
# plt.show()

import genprocess as gp
import genmodel as gm
import inference as i

# def compute_individual_VFE(pos_t, angles_t, mu_t, t_idx):
#     """ Function that computes the group free energy as a function of the positions, angles, and beliefs of a group of agents """

#     vel_t = jnp.concatenate([jnp.cos(angles_t).reshape(-1,1), jnp.sin(angles_t).reshape(-1,1)], axis = 1)
#     # sample observations from generative process
#     phi, all_dh_dr_self, empty_sectors_mask = gp.get_observations_special(pos_t, vel_t, genproc, t_idx)

#     # run hidden state inference 
#     infer_res, mu_traj = i.run_inference(phi, mu_t, empty_sectors_mask, genmodel, **meta_params['inference_params'])
#     mu_next, _ = infer_res

#     # compute and return variational free energy, summed over individuals
#     return gm.compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel)

def compute_group_VFE(pos_t, angles_t, mu_t, t_idx):
    """ Function that computes the group free energy as a function of the positions, angles, and beliefs of a group of agents """

    vel_t = jnp.concatenate([jnp.cos(angles_t).reshape(-1,1), jnp.sin(angles_t).reshape(-1,1)], axis = 1)
    # sample observations from generative process
    phi, all_dh_dr_self, empty_sectors_mask = gp.get_observations_special(pos_t, vel_t, genproc, t_idx)

    # run hidden state inference 
    infer_res, mu_traj = i.run_inference(phi, mu_t, empty_sectors_mask, genmodel, **meta_params['inference_params'])
    mu_next, _ = infer_res

    # compute and return variational free energy, summed over individuals
    return gm.compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel).sum()

angles_t = jnp.arctan2(vel_hist[t,:,1], vel_hist[t,:,0])
dFall_dtheta = jit(grad(compute_group_VFE, argnums = 1))
# dFall_dtheta_eval = dFall_dtheta(pos_hist[t], angles_t, mu_hist[t], t)


n_timesteps = len(genproc['t_axis'])
angles_hist = vmap(jnp.arctan2)(vel_hist[:,:,1], vel_hist[:,:,0])
theta_grads = vmap(dFall_dtheta)(pos_hist, angles_hist, mu_hist, jnp.arange(n_timesteps))

# # gradient_norms = jnp.linalg.norm(dFall_dvel_over_time, axis = 2)
# x_gradients = dFall_dvel_over_time[:,:,0]
# y_gradients = dFall_dvel_over_time[:,:,1]

color_vector = cm.Reds(np.linspace(0, 1, N)) # this goes from lightest reds (resp. blues) to darkest reds (resp. blues)

fig, ax = plt.subplots(1, 1, figsize=(10,8))
for t in range(n_timesteps-100, n_timesteps):
    sort_idx = jnp.argsort(jnp.absolute(theta_grads[t])) # default of `jnp.argsort` is lowest to highest, so "darkest" fish will be dots with highest dF_dtheta
    ax.scatter(pos_hist[t,sort_idx,0], pos_hist[t,sort_idx,1], s = 3.5, c = color_vector)

plt.show()

# dFindiv_dtheta = jacfwd(compute_individual_VFE, argnums = 1)
# J = dFindiv_dtheta(pos_hist[t], angles_t, mu_hist[t], t)

# import networkx as nx

# G = nx.DiGraph()

# for i in range(N):
#     for j in range(N):
#         G.add_edge(i,j,weight=jnp.absolute(J[j,i]))

# posdict = {i: pos_hist[t][i] for i in range(N)}
# weights = [G[u][v]['weight'] for u,v in G.edges()]
# nx.draw(G, pos = posdict, width = weights)

# plt.show()


# dFall_dtheta = jit(grad(compute_group_VFE, argnums = 1))
# dFall_dtheta_eval = dFall_dtheta(pos_hist[t], angles_t, mu_hist[t], t)

# n_timesteps = len(genproc['t_axis'])
# angles_hist = vmap(jnp.arctan2)(vel_hist[:,:,1], vel_hist[:,:,0])
# theta_grads = vmap(dFall_dtheta)(pos_hist, angles_hist, mu_hist, jnp.arange(n_timesteps))

# # # gradient_norms = jnp.linalg.norm(dFall_dvel_over_time, axis = 2)
# # x_gradients = dFall_dvel_over_time[:,:,0]
# # y_gradients = dFall_dvel_over_time[:,:,1]

# color_vector = cm.Reds(np.linspace(0, 1, N)) # this goes from lightest reds (resp. blues) to darkest reds (resp. blues)

# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# for t in range(n_timesteps-100, n_timesteps):
#     sort_idx = jnp.argsort(jnp.absolute(theta_grads[t])) # default of `jnp.argsort` is lowest to highest, so "darkest" fish will be dots with highest dF_dtheta
#     ax.scatter(pos_hist[t,sort_idx,0], pos_hist[t,sort_idx,1], s = 3.5, c = color_vector)
   
# # ax.plot(jnp.absolute(theta_grads))
# # fig, axes = plt.subplots(1, 2, figsize=(10,8))

# # for t in range(n_timesteps-100, n_timesteps):
# #     norm_sort_idx = jnp.argsort(x_gradients[t]) # default of `jnp.argsort` is lowest to highest, so "darkest" fish will be dots with highest ||dFall_dvel_i||
# #     axes[0].scatter(pos_hist[t,norm_sort_idx,0], pos_hist[t,norm_sort_idx,1], s = 3.5, c = color_vector)
# #     norm_sort_idx = jnp.argsort(y_gradients[t]) # default of `jnp.argsort` is lowest to highest, so "darkest" fish will be dots with highest ||dFall_dvel_i||
# #     axes[1].scatter(pos_hist[t,norm_sort_idx,0], pos_hist[t,norm_sort_idx,1], s = 3.5, c = color_vector)

# # fig, axes = plt.subplots(1, 2, figsize=(10,8))

# # for i in range(N):
# #     axes[0].plot(pos_hist[:,i,0], pos_hist[:,i,1])

# # for t in range(n_timesteps-100, n_timesteps):
# #     norm_sort_idx = jnp.argsort(gradient_norms[t]) # default of `jnp.argsort` is lowest to highest, so "darkest" fish will be dots with highest ||dFall_dvel_i||
# #     axes[1].scatter(pos_hist[t,norm_sort_idx,0], pos_hist[t,norm_sort_idx,1], s = 3.5, c = color_vector)

# plt.show()

# # print(gradient_norms.shape)

# # plt.plot(gradient_norms)
# # plt.show()
# # from jax_md import space
# # from jax_md.space import map_product, distance

# # displacement_fn, _ = space.free()

# # vmapped_displacement_fn = map_product(displacement_fn)

# # def fn_to_diff(pos, vel):
# #     within_sector_idx, distance_matrix, n2n_vecs = gp.compute_visual_neighbours(pos, vel, genproc['R_starts'], genproc['R_ends'], genproc['dist_thr'])
# #     h = gp.compute_h_per_sector(within_sector_idx, distance_matrix)
# #     return h.sum()
# # dfdposvel = grad(fn_to_diff, argnums = (0,1))
# # output = dfdposvel(pos_hist[t], vel_hist[t])
# # print(output[0])
# # print(output[1])

# # get h (first order observations)
# # h = gp.compute_h_per_sector(within_sector_idx, distance_matrix)
# # print(h)
# # dFall_dposvel = grad(compute_group_VFE, argnums = (0, 1, 2))
# # output = dFall_dposvel(pos_hist[t], vel_hist[t], mu_hist[t], t)
# # print(output[0])
# # print(output[1])
# # print(output[2])

# # Getting Nan gradients from the lines above, so let's go over the steps of compute_group_VFE and see where the NaNs arise in the computation graph

# # sample observations from generative process
# # phi, all_dh_dr_self, empty_sectors_mask = gp.get_observations(pos_hist[t], vel_hist[t], genproc, t)

# # run hidden state inference 
# # infer_res, mu_traj = i.run_inference(phi, mu_hist[t], empty_sectors_mask, genmodel, **meta_params['inference_params'])
# # mu_next, _ = infer_res



# # F_all = gm.compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel).sum()
# # def compute_summed_vfe(pos_t, vel_t, mu_t, t_idx):

# #     # THIS IS WHERE THE NANS OCCUR, NOT SURPRISINGLY
# #     phi, all_dh_dr_self, empty_sectors_mask = gp.get_observations(pos_t, vel_t, genproc, t_idx)

# #     # run hidden state inference 
# #     infer_res, mu_traj = i.run_inference(phi, mu_t, empty_sectors_mask, genmodel, **meta_params['inference_params'])
# #     mu_next, _ = infer_res

# #     return gm.compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel).sum()

# # dFall_dmposvel = jit(grad(compute_summed_vfe, argnums = (0, 1, 2)))
# # output = dFall_dmposvel(pos_hist[t], vel_hist[t], mu_hist[t], t)
# # print(output[0])
# # print(output[1])
# # print(output[2])

# ## So now, given that we know the NaN gradients arise in `gp.get_observations(...)`, let's move onto figuring out where that happens exactly
# # STEP 1: start at the end of gp.get_observations, the function called `sensory_samples_multi_order`

# # from genprocess import compute_visual_neighbours, compute_h_per_sector, compute_hprime_per_sector, sensory_samples_multi_order

# # # compute visual neighbourhoods 
# # within_sector_idx, distance_matrix, n2n_vecs = compute_visual_neighbours(pos_hist[t], vel_hist[t], genproc['R_starts'], genproc['R_ends'], genproc['dist_thr'])

# # # get h (first order observations)
# # h = compute_h_per_sector(within_sector_idx, distance_matrix)

# # # get hprime (velocity of observations)
# # hprime, all_dh_dr_self = compute_hprime_per_sector(within_sector_idx, pos_hist[t], vel_hist[t], n2n_vecs)

# # # aggregate hidden states
# # x_tilde_gp = [h, hprime]

# # # sample observations
# # # phi, empty_sectors_mask = sensory_samples_multi_order(x_tilde_gp, genproc['sensory_noise'][t], genproc['sensory_transform'], genproc['grad_sensory_transform'], remove_zeros=True)

# # from jax import jacfwd
# # from functools import partial

# # def get_obs_fn(pos, vel):
    
# #     # get hprime (velocity of observations)
# #     hprime, all_dh_dr_self = compute_hprime_per_sector(within_sector_idx, pos, vel, n2n_vecs)

# #     return hprime.sum() # the gradients of hprime.sum() with respect to `vel` is NaNs
# #     # return all_dh_dr_self.sum() # this works (zero-gradients, but not NaNs)
# #     # aggregate hidden states
# #     # x_tilde_gp = [h, hprime]

# #     # sample observations
# #     # phi, empty_sectors_mask = sensory_samples_multi_order(x, genproc['sensory_noise'][t], genproc['sensory_transform'], genproc['grad_sensory_transform'], remove_zeros=True)
# #     # return empty_sectors_mask.astype(jnp.float32).sum()

# # output = grad(get_obs_fn, argnums = (0,1))(pos_hist[t], vel_hist[t])
# # # output = grad(get_obs_fn)(x_tilde_gp)
# # print(output)

# ## So now that we know hprime is the source of zero-grads, let's unpack compute_hprime_per_sector to find out where it's happening in the computation graph

# # expanded_wsect_idx = within_sector_idx[...,None] # add a lagging dimension to `within_sector_idx` to enable broadcasted multiplications
# # sector_r = expanded_wsect_idx * n2n_vecs[None, ...] # matrix of shape (n_sectors, N, N, D) where sector_r[i, j, k, :] contains the "sector vector" pointing from neighbour k to focal agent j within sector i i.e. pos[k,:] - pos[j,:]

# # # normalize all sector vectors to unit norm
# # sector_r /= jnp.linalg.norm(sector_r, axis = 3, keepdims=True)
# # sector_r = jnp.nan_to_num(sector_r)

# # all_dh_dr_others = sector_r / expanded_wsect_idx.sum(axis=2, keepdims = True)

# # # you can compute `all_dh_dr_others` first and then using it, compute `all_dh_dr_self` as follows (need to test):
# # all_dh_dr_self = -all_dh_dr_others.sum(axis=2) # gradient of the average sector-wise distance with respect to oneself

# # sector_v = expanded_wsect_idx * vel[None, None, ...] # matrix of shape (n_sectors, N, N, D) where sector_v[i, j, k, :] contains the "sector velocity" of neighbour k within sector `i` of focal agent j

# # from genprocess import compute_hprime_vectorized

# # # print(sector_v)
# # # hprime = compute_hprime_vectorized(all_dh_dr_self, vel, all_dh_dr_others, sector_v)
# # def fn_to_diff(vel):
# #     return compute_hprime_vectorized(jnp.nan_to_num(all_dh_dr_self), vel, jnp.nan_to_num(all_dh_dr_others), sector_v).sum()

# # output = grad(fn_to_diff)(vel_hist[t])
# # print(output)


# # from jax import jacfwd
# # from functools import partial

# # def get_obs_fn(pos_t, vel_t):
# #     phi = gp.get_observations(pos_t, vel_t, genproc, t)
# #     return phi
# # # get_obs_fn = partial(gp.get_observations, genproc=genproc, t_idx = t)
# # jacob_obs_fn = jacfwd(get_obs_fn, argnums = (0,1))
# # out = jacob_obs_fn(pos_hist[t], vel_hist[t])
# # print(type(out))
# # phi, all_dh_dr_self, empty_sectors_mask = gp.get_observations(pos_t, vel_t, genproc, t_idx)


# # plot the history of positions over time

# # fig, ax = plt.subplots(figsize=(10,8))
# # for i in range(N):
# #     ax.plot(pos_hist[:,i,0], pos_hist[:,i,1])

# # plt.show()



