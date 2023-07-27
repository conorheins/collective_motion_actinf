from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from funcy import merge
import math
from scipy import stats

from utils import get_default_inits, make_single_timestep_fn, make_single_timestep_fn_nolearning, initialize_meta_params

from learning import make_dfdparams_fn, reparameterize
from genprocess import init_gen_process, change_noise_variance, compute_sector_dists_over_time
from genmodel import init_genmodel
from matplotlib import pyplot as plt

key = random.PRNGKey(4)

T, dt, N, D = 100, 0.01, 75, 2
burn_in_time= 1000
noise_change_t = 5000

z_hprime_var = 1.0

init_dict = get_default_inits(N, T, dt)
init_dict['z_h'] = 1.0
init_dict['z_hprime'] = z_hprime_var

pos, vel, genproc, new_key = init_gen_process(key, init_dict)

# alters the variance of sensory noise halfway through the simulation
noise_change_scalar = math.sqrt(10.0)
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

dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)
def step_fn(carry, t):
    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)
    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['s_z'], F)
init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))
position_history, velocity_history, sz_history, F_history = history[0], history[1], history[3], history[4]

sz_median = jnp.median(sz_history[burn_in_time:].flatten()) 
sz_mad = stats.median_abs_deviation(sz_history[burn_in_time:].flatten(), scale=1)
sz_bounds = [sz_median - 5*sz_mad, sz_median + 5*sz_mad]

fig, axes = plt.subplots(1,3,figsize=(12,6))
good_fish_ids = []
for n_idx in range(N):
    # check whether learned parameters within stable range
    if jnp.all((sz_history[burn_in_time:,n_idx] > sz_bounds[0])) and jnp.all(sz_history[burn_in_time:,n_idx] < sz_bounds[1]):
        good_fish_ids.append(n_idx)

good_fish_ids = jnp.array(good_fish_ids)

n_timesteps = len(genproc['t_axis'])

for i in good_fish_ids:
    axes[0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    axes[0].plot(position_history[noise_change_t:n_timesteps,i,0], position_history[noise_change_t:n_timesteps,i,1], c = 'r')
axes[0].set_title('Trajectories')

axes[1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),sz_history[burn_in_time:noise_change_t,good_fish_ids], c = 'b', label = 'before smoothness change')
axes[1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), sz_history[noise_change_t:,good_fish_ids], c = 'r', label = 'after smoothness change')
axes[1].set_title('Beliefs about smoothness')

axes[2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),F_history[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before smoothness change')
axes[2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), F_history[noise_change_t:,good_fish_ids].mean(-1), c = 'r', label = 'after smoothness change')
axes[2].set_title('Average surprise')

plt.savefig('/Users/conor/Documents/Active inference and collective motion/paper_figures/withlearning_smoothness_change.png', dpi = 325)

single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)
def step_fn(carry, t):
    pos_past, vel_past, mu_past = carry
    pos, vel, mu, F = single_timestep(pos_past, vel_past, mu_past, t)
    return (pos, vel, mu), (pos, vel, mu, F)
init_state = (pos, vel, mu)
_, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))
position_history, velocity_history, F_history = history[0], history[1], history[3]

fig, axes = plt.subplots(1,2,figsize=(12,6))
good_fish_ids = jnp.arange(N)

n_timesteps = len(genproc['t_axis'])

for i in good_fish_ids:
    axes[0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    axes[0].plot(position_history[noise_change_t:n_timesteps,i,0], position_history[noise_change_t:n_timesteps,i,1], c = 'r')
axes[0].set_title('Trajectories')

axes[1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),F_history[burn_in_time:noise_change_t,good_fish_ids].mean(-1), c = 'b', label = 'before smoothness change')
axes[1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), F_history[noise_change_t:,good_fish_ids].mean(-1), c = 'r', label = 'after smoothness change')
axes[1].set_title('Average surprise')

plt.savefig('/Users/conor/Documents/Active inference and collective motion/paper_figures/nolearning_smoothness_change.png', dpi = 325)


# if len(good_fish_ids) < 2:
#     print(f'Warning, trial {trial_i}, level {ii} with zhprime variance {zh_prime_var_i} has less than 2 good ids...\n')
#     try:
#         learned_sz_vals[ii,trial_i] = sz_history[burn_in_time:,good_fish_ids].mean() # average learned value of the s_z parameter, averaged across agents and timesteps
#         h_hist = compute_sector_dists_over_time(position_history, velocity_history, genproc)
#         average_nn_dist[ii,trial_i] = sz_history[burn_in_time:,good_fish_ids].mean() # average learned value of the s_z parameter, averaged across agents and timesteps
#     except:
#         learned_sz_vals[ii,trial_i] = np.nan
#         average_nn_dist[ii,trial_i] = np.nan
# else:
#     learned_sz_vals[ii,trial_i] = sz_history[burn_in_time:,good_fish_ids].mean(-1).mean() # average learned value of the s_z parameter, averaged across agents and timesteps
#     h_hist = compute_sector_dists_over_time(position_history, velocity_history, genproc)
#     average_nn_dist[ii,trial_i] = jnp.nanmean(h_hist[burn_in_time:,:,good_fish_ids],axis=1).mean()

