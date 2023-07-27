from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from funcy import merge
import math

from utils import get_default_inits, make_single_timestep_fn, make_single_timestep_fn_nolearning, initialize_meta_params

from learning import make_dfdparams_fn, reparameterize
from genprocess import init_gen_process, change_noise_variance, compute_sector_dists_over_time
from genmodel import init_genmodel
from matplotlib import pyplot as plt

key = random.PRNGKey(5)

T, dt, N, D = 200, 0.01, 75, 2
# T, dt, N, D = 40, 0.01, 20, 2

init_dict = get_default_inits(N, T, dt)
# init_dict['z_h'] = 1.0 # variance of additive observation noise on first order hidden states
# init_dict['z_hprime'] = 0.5 # variance of additive observation noise on second order hidden states ("velocity" observations)
init_dict = merge(init_dict, {'posvel_init': {'pos_x_bounds': [-3., 3.],
                                        'pos_y_bounds': [-3., 3.],
                                        'vel_x_bounds': [-1., 1.],
                                        'vel_x_bounds': [-1., 1.],
                                        }})
pos, vel, genproc, new_key = init_gen_process(key, init_dict)

# alters the variance of sensory noise halfway through the simulation
noise_change_t = int((len(genproc['t_axis'])/2))
noise_change_scalar = 10.0
genproc['sensory_noise'] = change_noise_variance(genproc['sensory_noise'], noise_change_t, noise_change_scalar, do_idx=1) # this only scales the variance for the embedding orders given by `do_idx`

genmodel = init_genmodel(init_dict)

meta_params = initialize_meta_params(infer_lr = 0.1, 
                                    nsteps_infer = 1, 
                                    action_lr = 0.1, 
                                    nsteps_action = 1, 
                                    learning_lr = 0.001, 
                                    nsteps_learning = 1, 
                                    normalize_v = True
                                    )

## Parameterization specific to learning s_z
_, smoothness_key = random.split(new_key)

average_s_z = 1.0
lower_bound, upper_bound = average_s_z - 0.1, average_s_z + 0.1
s_z_all = random.uniform(smoothness_key, minval = lower_bound, maxval = upper_bound, shape = (N,)) # sample a different sensory smoothness for every agent

Pi_z_spatial = init_dict['pi_z_spatial'] * jnp.eye(genmodel['ns_phi'])
def parameterize_Pi_2do(s_z):
    """
    Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
    """
    Pi_z_temporal = jnp.diag(jnp.array([1.0, 2 * s_z**2]))
    return jnp.kron(Pi_z_temporal, Pi_z_spatial)

preparams = {'s_z': s_z_all}
parameterization_mapping = {'s_z': {'to_arg_name': 'Pi_z', 'fn': parameterize_Pi_2do}}

initial_learnable_params = vmap(reparameterize, (0, None))(preparams, parameterization_mapping)
genmodel = merge(genmodel, initial_learnable_params)

mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

""" First, do the version without learning """
single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)
def step_fn(carry, t):
    pos_past, vel_past, mu_past = carry
    pos, vel, mu, F = single_timestep(pos_past, vel_past, mu_past, t)
    return (pos, vel, mu), (pos, vel, mu, F)
init_state = (pos, vel, mu)
_, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))
position_history, velocity_history, F_history = history[0], history[1], history[3]

n_timesteps = len(genproc['t_axis'])
burn_in_time= 1250

# get the centroid at the end of the simulation
centroid_noL = jnp.mean(position_history[-1], axis=0)
# get the euclidean distance between each agent's position at the end of the simulation and the centroid
dists_noL = jnp.linalg.norm(position_history[-1] - centroid_noL, axis=1)
# get the indices of the agents whose distance from the centroid is less than 0.5
good_fish_ids = jnp.where(dists_noL < 8.0)[0]

# h_hist = compute_sector_dists_over_time(position_history, velocity_history, genproc)
# average_nn_dists = jnp.nanmean(h_hist,axis=1)


fig, axes = plt.subplots(1,3, figsize=(14,6))
blues, reds = plt.cm.Blues(jnp.linspace(0.3, 1, 10)), plt.cm.Reds(jnp.linspace(0.3, 1, 10))

for i in good_fish_ids:
    axes[0].plot(position_history[burn_in_time:noise_change_t,i,0], position_history[burn_in_time:noise_change_t,i,1], c = blues[2])
    axes[0].plot(position_history[noise_change_t:n_timesteps,i,0], position_history[noise_change_t:n_timesteps,i,1], c = blues[8])

average_F = F_history[:,good_fish_ids].mean(-1)
average_F_relative = average_F - average_F[burn_in_time]   

axes[2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),average_F_relative[burn_in_time:noise_change_t], c = blues[2], label = 'before noise change')
axes[2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), average_F_relative[noise_change_t:n_timesteps], c = blues[8], label = 'after noise change')


# fig, axes = plt.subplots(2, 3, figsize=(10,8))
# for i in range(N):
#     axes[0,0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
#     axes[0,0].plot(position_history[noise_change_t:n_timesteps,i,0], position_history[noise_change_t:n_timesteps,i,1], c = 'r')

# axes[0,1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),jnp.nanmean(average_nn_dists[burn_in_time:noise_change_t], axis = 1), c = 'b', label = 'before noise change')
# axes[0,1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), jnp.nanmean(average_nn_dists[noise_change_t:n_timesteps], axis = 1), c = 'r', label = 'after noise change')
# axes[0,2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),F_history[burn_in_time:noise_change_t].mean(-1), c = 'b', label = 'before noise change')
# axes[0,2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), F_history[noise_change_t:n_timesteps].mean(-1), c = 'r', label = 'after noise change')


dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)
def step_fn(carry, t):
    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)
    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['s_z'], F)
init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))
position_history, velocity_history, sz_history, F_history = history[0], history[1], history[3], history[4]


# get the centroid at the end of the simulation
centroid_L = jnp.mean(position_history[-1], axis=0)
# get the euclidean distance between each agent's position at the end of the simulation and the centroid
dists_L = jnp.linalg.norm(position_history[-1] - centroid_L, axis=1)
# get the indices of the agents whose distance from the centroid is less than 0.5
good_fish_ids = jnp.where(dists_L < 8.0)[0]

# good_fish_ids = []
# for n_idx in range(N):
#     if jnp.all(sz_history[noise_change_t:,n_idx] < 5.0):
#         good_fish_ids.append(n_idx)
# good_fish_ids = jnp.array(good_fish_ids)

for i in good_fish_ids:
    axes[1].plot(position_history[burn_in_time:noise_change_t,i,0] + 10., position_history[burn_in_time:noise_change_t,i,1] + 10., c = reds[2])
    axes[1].plot(position_history[noise_change_t:n_timesteps,i,0] + 10., position_history[noise_change_t:n_timesteps,i,1]+ 10., c = reds[8])

average_F = F_history[:,good_fish_ids].mean(-1)
average_F_relative = average_F - average_F[burn_in_time]   

axes[2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),average_F_relative[burn_in_time:noise_change_t], c = reds[2], alpha=0.5)
axes[2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), average_F_relative[noise_change_t:n_timesteps], c = reds[8], alpha=0.5)

plt.savefig('learning_nolearning_comparison_noisechange.png', dpi =325)


# h_hist = compute_sector_dists_over_time(position_history, velocity_history, genproc)
# average_nn_dists = jnp.nanmean(h_hist,axis=1)

# for i in good_fish_ids:
#     axes[1,0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
#     axes[1,0].plot(position_history[noise_change_t:n_timesteps,i,0], position_history[noise_change_t:n_timesteps,i,1], c = 'r')

# axes[1,1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),average_nn_dists[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before noise change')
# axes[1,1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), average_nn_dists[noise_change_t:n_timesteps,good_fish_ids].mean(-1), c = 'r', label = 'after noise change')
# axes[1,2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),F_history[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before smoothness change')
# axes[1,2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), F_history[noise_change_t:n_timesteps,good_fish_ids].mean(-1), c = 'r', label = 'after smoothness change')
# # axes[1,2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),sz_history[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before smoothness change')
# # axes[1,2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), sz_history[noise_change_t:n_timesteps,good_fish_ids].mean(-1), c = 'r', label = 'after smoothness change')
# plt.show()