from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from funcy import merge
import math

from utils import make_single_timestep_fn, make_single_timestep_fn_nolearning, initialize_meta_params

from learning import make_dfdparams_fn, reparameterize
from genprocess import init_gen_process, change_noise_variance, compute_sector_dists_over_time
from genmodel import init_genmodel
from matplotlib import pyplot as plt

key = random.PRNGKey(4)

T, dt, N, D = 200, 0.01, 75, 2

initialization_dict = {'N': N,
                        'posvel_init': {'pos_x_bounds': [-1., 1.],
                                        'pos_y_bounds': [-1., 1.],
                                        'vel_x_bounds': [-1., 1.],
                                        'vel_x_bounds': [-1., 1.],
                                        },
                        'T': T, # total length of simulation (in seconds)
                        'dt': dt, # duration of integration timestep for stochastic integration (in seconds)
                        'sector_angles': [120., 60., 0., 360. - 60., 360. - 120.], # angles of visual sectors
                        'ns_x': 4, # dimensionality of hidden states
                        'ndo_x': 3, # number of dynamical orders of hidden states
                        'ns_phi': 4, # dimensionality of observations
                        'ndo_phi': 2, # number of dynamical orders of observations
                        'dist_thr': 5.0, # cut-off within which neighbouring agents are detected
                        'z_h': 1.0,      # variance of additive observation noise on first order hidden states
                        'z_hprime': 0.5, # variance of additive observation noise on second order hidden states ("velocity" observations)
                        'z_action': 0.01, # variance of movement/action (additive noise onto x/y components of velocity vector during stochastic integration),
                        'alpha': 0.5,   # strength of flow function (the decay coefficient in case of independent dimensions)
                        'eta': 1.,       # the fixed point of the flow function
                        'pi_z_spatial': 1.0, # the spatial variance of the sensory precision
                        'pi_w_spatial': 1.0, # the spatial variance of the model or "process" precision
                        's_z': 1.0,          # the assumed smoothness (temporal autocorrelation) of sensory fluctuations
                        's_w': 1.0           # the assumed smoothness (temporal autocorrelation) of process fluctuations
}
pos, vel, genproc, new_key = init_gen_process(key, initialization_dict)

# alters the variance of sensory noise halfway through the simulation
noise_change_t = int((len(genproc['t_axis'])/2))
noise_change_scalar = 10.0
genproc['sensory_noise'] = change_noise_variance(genproc['sensory_noise'], noise_change_t, noise_change_scalar, do_idx=1) # this only scales the variance for the embedding orders given by `do_idx`

genmodel = init_genmodel(initialization_dict)

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

Pi_z_spatial = initialization_dict['pi_z_spatial'] * jnp.eye(genmodel['ns_phi'])
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
burn_in_time= 1000

h_hist = compute_sector_dists_over_time(position_history, velocity_history, genproc)
average_nn_dists = jnp.nanmean(h_hist,axis=1)

fig, axes = plt.subplots(2, 3, figsize=(10,8))
for i in range(N):
    axes[0,0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    axes[0,0].plot(position_history[noise_change_t:n_timesteps,i,0], position_history[noise_change_t:n_timesteps,i,1], c = 'r')

axes[0,1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),jnp.nanmean(average_nn_dists[burn_in_time:noise_change_t], axis = 1), c = 'b', label = 'before noise change')
axes[0,1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), jnp.nanmean(average_nn_dists[noise_change_t:n_timesteps], axis = 1), c = 'r', label = 'after noise change')
axes[0,2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),F_history[burn_in_time:noise_change_t].mean(-1), c = 'b', label = 'before noise change')
axes[0,2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), F_history[noise_change_t:n_timesteps].mean(-1), c = 'r', label = 'after noise change')


dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)
def step_fn(carry, t):
    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)
    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['s_z'], F)
init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))
position_history, velocity_history, sz_history, F_history = history[0], history[1], history[3], history[4]

good_fish_ids = []
for n_idx in range(N):
    if jnp.all(sz_history[noise_change_t:,n_idx] < 5.0):
        good_fish_ids.append(n_idx)
good_fish_ids = jnp.array(good_fish_ids)

h_hist = compute_sector_dists_over_time(position_history, velocity_history, genproc)
average_nn_dists = jnp.nanmean(h_hist,axis=1)

for i in good_fish_ids:
    axes[1,0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    axes[1,0].plot(position_history[noise_change_t:n_timesteps,i,0], position_history[noise_change_t:n_timesteps,i,1], c = 'r')

axes[1,1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),average_nn_dists[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before noise change')
axes[1,1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), average_nn_dists[noise_change_t:n_timesteps,good_fish_ids].mean(-1), c = 'r', label = 'after noise change')
axes[1,2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),F_history[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before smoothness change')
axes[1,2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), F_history[noise_change_t:n_timesteps,good_fish_ids].mean(-1), c = 'r', label = 'after smoothness change')
# axes[1,2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),sz_history[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before smoothness change')
# axes[1,2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), sz_history[noise_change_t:n_timesteps,good_fish_ids].mean(-1), c = 'r', label = 'after smoothness change')
plt.show()