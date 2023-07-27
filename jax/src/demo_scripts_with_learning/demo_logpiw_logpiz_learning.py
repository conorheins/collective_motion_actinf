from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from scipy import stats


from utils import make_single_timestep_fn, initialize_meta_params

from learning import make_dfdparams_fn, reparameterize
from funcy import merge
from genprocess import init_gen_process, change_noise_variance, compute_dist_matrices_over_time, compute_sector_dists_over_time
from genmodel import init_genmodel, create_temporal_precisions
from matplotlib import pyplot as plt

key = random.PRNGKey(3)

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
                        'eta': 1.,       # the fixed point of the flow function
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
# _, Pi_z_temporal = create_temporal_precisions(truncation_order=genmodel['ndo_phi'], smoothness=initialization_dict['s_z'])   # incorrect since you're using covariance, but weirdly more stable...?

# Pi_z_temporal = jnp.eye(genmodel['ndo_phi'])
def parameterize_Pi_z(logpi_z_spatial):
    """
    Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
    """
    Pi_z_spatial = jnp.exp(logpi_z_spatial) * jnp.eye(genmodel['ns_phi'])
    return jnp.kron(Pi_z_temporal, Pi_z_spatial)

Pi_w_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_x'], smoothness=initialization_dict['s_w']) # technically correct, but need to decrease smoothness s_w to make it look ok
# _, Pi_w_temporal = create_temporal_precisions(truncation_order=genmodel['ndo_x'], smoothness=initialization_dict['s_w']) # incorrect since you're using covariance, but weirdly more stable...?

# Pi_w_temporal = jnp.eye(genmodel['ndo_x'])
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

dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)

def step_fn(carry, t):

    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)

    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['logpiz_spatial'],preparams['logpiw_spatial'], F)

init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))

position_history, velocity_history = history[0], history[1]

n_timesteps = len(genproc['t_axis'])

fig, axes = plt.subplots(1, 3, figsize=(10,8))
for i in range(N):
    axes[0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    axes[0].plot(position_history[noise_change_t:n_timesteps,i,0], position_history[noise_change_t:n_timesteps,i,1], c = 'r')

logpizspatial_history, logpiwspatial_history = history[3], history[4]

z_variance_history = 1. / jnp.exp(logpizspatial_history)
w_variance_history = 1. / jnp.exp(logpiwspatial_history)

burn_in_time= 1000

z_var_median = jnp.median(z_variance_history[burn_in_time:].flatten()) 
z_var_mad = stats.median_abs_deviation(z_variance_history[burn_in_time:].flatten(), scale=1)
z_var_bounds = [z_var_median - 5*z_var_mad, z_var_median + 5*z_var_mad]

w_var_median = jnp.median(w_variance_history[burn_in_time:].flatten()) 
w_var_mad = stats.median_abs_deviation(w_variance_history[burn_in_time:].flatten(), scale=1)
w_var_bounds = [w_var_median - 5*w_var_mad, w_var_median + 5*w_var_mad]

good_fish_ids = []
for n_idx in range(N):

    stable_z_var = jnp.all((z_variance_history[burn_in_time:,n_idx] > z_var_bounds[0])) and jnp.all(z_variance_history[burn_in_time:,n_idx] < z_var_bounds[1])

    stable_w_var = jnp.all((w_variance_history[burn_in_time:,n_idx] > w_var_bounds[0])) and jnp.all(w_variance_history[burn_in_time:,n_idx] < w_var_bounds[1])

    if stable_z_var and stable_w_var:
        good_fish_ids.append(n_idx)

good_fish_ids = jnp.array(good_fish_ids)

# dist_matrices_t = compute_dist_matrices_over_time(position_history[:,good_fish_ids,:])

# def compute_average_distance(dist_matrix):
#     """
#     Compute average nearest neighbour distance usign the N x N distance matrix of neighbouring positions
#     """
#     ind_x, ind_y = jnp.tril_indices_from(dist_matrix, k = -1)
#     return jnp.median(dist_matrix[ind_x, ind_y])

# average_nn_dists = vmap(compute_average_distance)(dist_matrices_t)

# h_hist = compute_sector_dists_over_time(position_history, velocity_history, genproc)
# average_nn_dists = jnp.nanmean(h_hist[:,:,good_fish_ids],axis=1)

# axes[1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),average_nn_dists[burn_in_time:noise_change_t], c = 'b', label = 'before noise change')
# axes[1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), average_nn_dists[noise_change_t:n_timesteps], c = 'r', label = 'after noise change')

axes[1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),z_variance_history[burn_in_time:noise_change_t, good_fish_ids], c = 'b', label = 'before noise change')
axes[1].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), z_variance_history[noise_change_t:n_timesteps, good_fish_ids], c = 'r', label = 'after noise change')

axes[2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t), w_variance_history[burn_in_time:noise_change_t, good_fish_ids], c = 'b', label = 'before noise change')
axes[2].plot(jnp.arange(start = noise_change_t, stop = n_timesteps), w_variance_history[noise_change_t:n_timesteps, good_fish_ids], c = 'r', label = 'after noise change')
plt.show()
