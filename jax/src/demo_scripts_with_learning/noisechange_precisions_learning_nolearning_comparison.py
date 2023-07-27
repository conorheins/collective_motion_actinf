from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from scipy import stats
from funcy import merge

from utils import make_single_timestep_fn, make_single_timestep_fn_nolearning, initialize_meta_params

from learning import make_dfdparams_fn, reparameterize
from genprocess import init_gen_process, change_noise_variance, compute_sector_dists_over_time
from genmodel import init_genmodel, create_temporal_precisions
from matplotlib import pyplot as plt

key = random.PRNGKey(4)

T, dt, N, D = 100, 0.01, 25, 2

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
                        'z_h': 0.1,      # variance of additive observation noise on first order hidden states
                        'z_hprime': 0.1, # variance of additive observation noise on second order hidden states ("velocity" observations)
                        'z_action': 0.01, # variance of movement/action (additive noise onto x/y components of velocity vector during stochastic integration),
                        'alpha': 0.5,   # strength of flow function (the decay coefficient in case of independent dimensions)
                        'eta': 1.0,       # the fixed point of the flow function
                        'pi_z_spatial': 1.0, # the spatial variance of the sensory precision
                        'pi_w_spatial': 1.0, # the spatial variance of the model or "process" precision
                        's_z': 0.5,          # the assumed smoothness (temporal autocorrelation) of sensory fluctuations
                        's_w': 0.5           # the assumed smoothness (temporal autocorrelation) of process fluctuations
}
pos, vel, genproc, new_key = init_gen_process(key, initialization_dict)

# alters the variance of sensory noise halfway through the simulation
noise_change_t = int(0.5 * len(genproc['t_axis']))
noise_change_scalar = 5.0
genproc['sensory_noise'] = change_noise_variance(genproc['sensory_noise'], noise_change_t, noise_change_scalar)

genmodel = init_genmodel(initialization_dict)

meta_params = initialize_meta_params(infer_lr = 0.1, 
                                    nsteps_infer = 1, 
                                    action_lr = 0.1, 
                                    nsteps_action = 1, 
                                    learning_lr = 0.001, 
                                    nsteps_learning = 1, 
                                    normalize_v = True
                                    )

## Parameterization specific to learning f_params
piw_key, piz_key = random.split(new_key)

logpiz_spatial_all = 0.5 * random.normal(piz_key, shape = (N,)) # sample a different sensory (log) spatial precision for every agent
logpiw_spatial_all = 0.5 * random.normal(piw_key, shape = (N,)) # sample a different sensory (log) spatial precision for every agent

Pi_z_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_phi'], smoothness=initialization_dict['s_z']) # technically correct, but need to decrease smoothness s_z to make it look ok
def parameterize_Pi_z(logpi_z_spatial):
    """
    Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
    """
    Pi_z_spatial = jnp.exp(logpi_z_spatial) * jnp.eye(genmodel['ns_phi'])
    return jnp.kron(Pi_z_temporal, Pi_z_spatial)

Pi_w_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_x'], smoothness=initialization_dict['s_w']) # technically correct, but need to decrease smoothness s_w to make it look ok
def parameterize_Pi_w(logpi_w_spatial):
    """
    Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
    """
    Pi_w_spatial = jnp.exp(logpi_w_spatial) * jnp.eye(genmodel['ns_x'])
    return jnp.kron(Pi_w_temporal, Pi_w_spatial)


preparams = {'logpiz_spatial': logpiz_spatial_all, 'logpiw_spatial': logpiw_spatial_all}
parameterization_mapping = {'logpiz_spatial': {'to_arg_name': 'Pi_z', 'fn': parameterize_Pi_z},
                            'logpiw_spatial': {'to_arg_name': 'Pi_w', 'fn': parameterize_Pi_w}
                            }

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
mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T
def step_fn(carry, t):
    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)
    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['logpiz_spatial'],preparams['logpiw_spatial'], F)
init_state = (pos, vel, mu, preparams)
_, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))
position_history, velocity_history, F_history = history[0], history[1], history[5]

good_fish_ids = []
for n_idx in range(N):

    if jnp.all(F_history[burn_in_time:,n_idx] < 100.0):
        good_fish_ids.append(n_idx)

good_fish_ids = jnp.array(good_fish_ids)

h_hist = compute_sector_dists_over_time(position_history, velocity_history, genproc)
average_nn_dists = jnp.nanmean(h_hist,axis=1)

for i in good_fish_ids:
    axes[1,0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    axes[1,0].plot(position_history[noise_change_t:n_timesteps,i,0], position_history[noise_change_t:n_timesteps,i,1], c = 'r')

# axes[1,1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),average_nn_dists[burn_in_time:noise_change_t,good_fish_ids], c = 'b', label = 'before noise change')
# axes[1,1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), average_nn_dists[noise_change_t:n_timesteps,good_fish_ids], c = 'r', label = 'after noise change')

axes[1,1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),average_nn_dists[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before noise change')
axes[1,1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), average_nn_dists[noise_change_t:n_timesteps,good_fish_ids].mean(-1), c = 'r', label = 'after noise change')


axes[1,2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),F_history[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before noise change')
axes[1,2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), F_history[noise_change_t:n_timesteps,good_fish_ids].mean(-1), c = 'r', label = 'after noise change')

plt.show()
