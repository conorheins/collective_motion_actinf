import jax
from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from scipy import stats
from utils import make_single_timestep_fn, make_single_timestep_fn_nolearning, initialize_meta_params

from learning import make_dfdparams_fn
from genprocess import init_gen_process, compute_sector_dists_over_time
from genprocess import geometry as geo
from genmodel import init_genmodel, parameterize_A0_no_coupling
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

# this seed / group size / time length makes a nice figure
key = random.PRNGKey(2)
T, dt, N, D = 200, 0.01, 100, 2

cpus = jax.devices("cpu")

initialization_dict = {'N': N,
                        'posvel_init': {'pos_x_bounds': [-10., 10.],
                                        'pos_y_bounds': [-10., 10.],
                                        'vel_x_bounds': [-1., 1.],
                                        'vel_x_bounds': [-1., 1.],
                                        },
                        'T': T, # total length of simulation (in seconds)
                        'dt': dt, # duration of integration timestep for stochastic integration (in seconds)
                        'sector_angles': [120.,60., 0., 360.-60.,360.-120.], # angles of visual sectors
                        'ns_x': 4, # dimensionality of hidden states
                        'ndo_x': 3, # number of dynamical orders of hidden states
                        'ns_phi': 4, # dimensionality of observations
                        'ndo_phi': 2, # number of dynamical orders of observations
                        'dist_thr': 5.0, # cut-off within which neighbouring agents are detected
                        'z_h': 0.1,      # variance of additive observation noise on first order hidden states
                        'z_hprime': 0.1, # variance of additive observation noise on second order hidden states ("velocity" observations)
                        'z_action': 0.001, # variance of movement/action (additive noise onto x/y components of velocity vector during stochastic integration),
                        'alpha': 0.5,   # strength of flow function (the decay coefficient in case of independent dimensions)
                        'eta': 1.,       # the fixed point of the flow function
                        'pi_z_spatial': 1.0, # the spatial variance of the sensory precision
                        'pi_w_spatial': 2.0, # the spatial variance of the model or "process" precision
                        's_z': 1.0,          # the assumed smoothness (temporal autocorrelation) of sensory fluctuations
                        's_w': 1.5           # the assumed smoothness (temporal autocorrelation) of process fluctuations
}
pos, vel, genproc, new_key = init_gen_process(key, initialization_dict)

genmodel = init_genmodel(initialization_dict)

variance_term = 0.5 * (-jnp.linalg.slogdet(genmodel['Pi_w'][0])[1] - jnp.linalg.slogdet(genmodel['Pi_z'][0])[1]) # variance term of the VFE

meta_params = initialize_meta_params(infer_lr = 0.1, 
                                    nsteps_infer = 1, 
                                    action_lr = 0.2, 
                                    nsteps_action = 1, 
                                    learning_lr = 0.001, 
                                    nsteps_learning = 1, 
                                    normalize_v = True
                                    )


## Parameterization specific to learning f_params
_, eta_key = random.split(new_key)

alpha_all = initialization_dict['alpha'] * jnp.ones((N,)) # sample a different baseline alpha for every agent
eta0_all = initialization_dict['eta'] + 0.1*random.normal(eta_key, shape = (N,1,genmodel['ns_x'])) # fix the baseline eta for every sector, for every agent

def parameterize_f_params(f_params_pre):
    """
    Parameterize tilde eta (the attracting point of the flow function) with preparams, stop gradient on parameters of the A0 matrix
    """
    f_params = {
        'tilde_A': jnp.stack(genmodel['ndo_x'] * [parameterize_A0_no_coupling(lax.stop_gradient(f_params_pre['alpha']), genmodel['ns_x'])]), 
        'tilde_eta': jnp.concatenate((f_params_pre['eta0'], jnp.zeros((genmodel['ndo_x'] -1, genmodel['ns_x'] )))) # added lax.stop_gradient here to stop eta from getting updated
    }
    return f_params

preparams = {'f_params_pre': {'alpha': alpha_all, 'eta0': eta0_all}}

parameterization_mapping = {'f_params_pre': {'to_arg_name': 'f_params', 'fn': parameterize_f_params} }

genmodel['f_params'] = vmap(parameterize_f_params)(preparams['f_params_pre'])

mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)
def step_fn(carry, t):
    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)
    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['f_params_pre']['eta0'], F)
n_timesteps = len(genproc['t_axis'])
init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(n_timesteps))
position_history, velocity_history, mu_history, eta0_history, F_history_wlearning = history[0], history[1], history[2], history[3], history[4]

h_hists = compute_sector_dists_over_time(jax.device_put(position_history, cpus[0]), jax.device_put(velocity_history, cpus[0]), genproc)

steady_state_time = int(0.5 * n_timesteps)
average_sector_h_wlearning = jnp.nanmean(h_hists[steady_state_time:], axis=0).T.flatten()
average_learned_eta = eta0_history[steady_state_time:,:,0,:].mean(axis=0).squeeze().flatten()
non_nan_idx1 = jnp.logical_not(jnp.isnan(average_sector_h_wlearning))

fig, axes = plt.subplots(2, 3, figsize=(12,8))
colors = cm.rainbow(np.linspace(0, 1, N))
from_end = 100

time_range = jnp.arange(n_timesteps-from_end, n_timesteps)
which_to_inspect = np.array(list(range(N)), dtype=int)
colors_subset = colors[which_to_inspect]
for i, c in zip(which_to_inspect, colors_subset):
    axes[0,0].plot(position_history[time_range,i,0], position_history[time_range,i,1], c=c)

axes[0,1].scatter(average_sector_h_wlearning[non_nan_idx1], average_learned_eta[non_nan_idx1])
print(f'Average free energy: {F_history_wlearning[steady_state_time:].mean() - variance_term}')

preparams = {'f_params_pre': {'alpha': alpha_all, 'eta0': eta0_all}}
parameterization_mapping = {'f_params_pre': {'to_arg_name': 'f_params', 'fn': parameterize_f_params} }

genmodel = init_genmodel(initialization_dict)
genmodel['f_params'] = vmap(parameterize_f_params)(preparams['f_params_pre'])

mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T
single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)
def step_fn(carry, t):
    pos_past, vel_past, mu_past = carry
    pos, vel, mu, F = single_timestep(pos_past, vel_past, mu_past, t)
    return (pos, vel, mu), (pos, vel, mu, F)
n_timesteps = len(genproc['t_axis'])
init_state = (pos, vel, mu)
out, history = lax.scan(step_fn, init_state, jnp.arange(n_timesteps))
position_history, velocity_history, mu_history, F_history_wolearning = history[0], history[1], history[2], history[3]

h_hists = compute_sector_dists_over_time(jax.device_put(position_history, cpus[0]), jax.device_put(velocity_history, cpus[0]), genproc)
steady_state_time = int(0.5 * n_timesteps)
average_sector_h_wolearning = jnp.nanmean(h_hists[steady_state_time:], axis=0).T.flatten()
non_nan_idx2 = jnp.logical_not(jnp.isnan(average_sector_h_wolearning))

which_to_inspect = np.array(list(range(N)), dtype=int)
colors_subset = colors[which_to_inspect]
for i, c in zip(which_to_inspect, colors_subset):
    axes[1,0].plot(position_history[time_range,i,0], position_history[time_range,i,1], c=c)

axes[1,1].scatter(average_sector_h_wolearning[non_nan_idx2], eta0_all.squeeze().flatten()[non_nan_idx2])
# axes[1,1].plot(F_history_wlearning[steady_state_time:].mean(-1)- variance_term)
# axes[1,1].plot(F_history_wolearning[steady_state_time:].mean(-1)- variance_term)

full_sector_h_samples = np.hstack( [average_sector_h_wlearning[non_nan_idx1], average_sector_h_wolearning[non_nan_idx2]] )
common_bins = np.histogram_bin_edges(full_sector_h_samples, bins = 15)

x_axis = jnp.linspace(common_bins[0], common_bins[-1], num = 50)

kernel1 = stats.gaussian_kde(average_sector_h_wlearning[non_nan_idx1], bw_method = 0.25)
axes[0,2].plot(x_axis, kernel1(x_axis))
kernel2 = stats.gaussian_kde(average_learned_eta[non_nan_idx1], bw_method = 0.25)
axes[0,2].plot(x_axis, kernel2(x_axis))

# axes[0,2].hist(average_sector_h_wlearning[non_nan_idx1], bins = common_bins)
# axes[0,2].hist(average_learned_eta[non_nan_idx1], bins = common_bins)

kernel1 = stats.gaussian_kde(average_sector_h_wolearning[non_nan_idx2], bw_method = 0.25)
axes[1,2].plot(x_axis, kernel1(x_axis))
kernel2 =stats. gaussian_kde(eta0_all.squeeze().flatten()[non_nan_idx2], bw_method = 0.25)
axes[1,2].plot(x_axis, kernel2(x_axis))

# axes[1,2].hist(average_sector_h_wolearning[non_nan_idx2], bins = common_bins)
# axes[1,2].hist(eta0_all.squeeze().flatten()[non_nan_idx2], bins = common_bins)

print(f'Average free energy: {F_history_wolearning[steady_state_time:].mean()- variance_term}')
plt.show()
