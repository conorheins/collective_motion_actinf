from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from funcy import merge
from functools import partial

from utils import make_single_timestep_fn, initialize_meta_params

from learning import make_dfdparams_fn, reparameterize
from genprocess import init_gen_process, compute_sector_dists_over_time
from genprocess import geometry as geo
from genmodel import init_genmodel, create_temporal_precisions, parameterize_A0_no_coupling
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

key = random.PRNGKey(201)

T, dt, N, D = 2000, 0.01, 3, 2

initialization_dict = {'N': N,
                        'posvel_init': {'pos_x_bounds': [-1., 1.],
                                        'pos_y_bounds': [-1., 1.],
                                        'vel_x_bounds': [-1., 1.],
                                        'vel_x_bounds': [-1., 1.],
                                        },
                        'T': T, # total length of simulation (in seconds)
                        'dt': dt, # duration of integration timestep for stochastic integration (in seconds)
                        'sector_angles': [120., 0., 360.-120.], # angles of visual sectors
                        'ns_x': 2, # dimensionality of hidden states
                        'ndo_x': 3, # number of dynamical orders of hidden states
                        'ns_phi': 2, # dimensionality of observations
                        'ndo_phi': 2, # number of dynamical orders of observations
                        'dist_thr': 5.0, # cut-off within which neighbouring agents are detected
                        'z_h': 0.1,      # variance of additive observation noise on first order hidden states
                        'z_hprime': 0.1, # variance of additive observation noise on second order hidden states ("velocity" observations)
                        'z_action': 0.001, # variance of movement/action (additive noise onto x/y components of velocity vector during stochastic integration),
                        'alpha': 0.3,   # strength of flow function (the decay coefficient in case of independent dimensions)
                        'eta': 1.,       # the fixed point of the flow function
                        'pi_z_spatial': 1.0, # the spatial variance of the sensory precision
                        'pi_w_spatial': 1.0, # the spatial variance of the model or "process" precision
                        's_z': 1.0,          # the assumed smoothness (temporal autocorrelation) of sensory fluctuations
                        's_w': 1.0           # the assumed smoothness (temporal autocorrelation) of process fluctuations
}
pos, vel, genproc, new_key = init_gen_process(key, initialization_dict)

agent1_pos = jnp.array([-.25, 0.0])
agent2_pos = 0.5 * jnp.array([jnp.sqrt(2), jnp.sqrt(2)])
agent3_pos = 0.5 * jnp.array([-jnp.sqrt(2), jnp.sqrt(2)])
# agent4_pos = jnp.array([0.1*jnp.sqrt(2), 0.1*jnp.sqrt(2)])
# pos = jnp.stack( [agent1_pos, agent2_pos, agent3_pos, agent4_pos], axis = 0)
pos = jnp.stack( [agent1_pos, agent2_pos, agent3_pos], axis = 0)

# agent1_vel = jnp.array([0.0, 1.0])
# agent2_vel = jnp.array([0.5, 1.0]) / jnp.linalg.norm(jnp.array([0.5, 1.0]))
# agent3_vel = jnp.array([0.0, 1.0])
# vel = jnp.stack( [agent1_vel, agent2_vel, agent3_vel], axis = 0)

vel = jnp.stack( N*[jnp.array([0.0, 1.0])], axis = 0).squeeze()

# compute visual neighbourhoods and local distances (can be used to initialize eta)
within_sector_idx, distance_matrix, _ = geo.compute_visual_neighbours(pos, vel, genproc['R_starts'], genproc['R_ends'], genproc['dist_thr'])
h = geo.compute_h_per_sector_keepNaNs(within_sector_idx, distance_matrix)

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

logpiz_spatial_all, logpiw_spatial_all = jnp.ones((N,2)), jnp.ones((N,2))

Pi_z_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_phi'], smoothness=initialization_dict['s_z']) # technically correct, but need to decrease smoothness s_z to make it look ok
Pi_w_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_x'], smoothness=initialization_dict['s_w']) # technically correct, but need to decrease smoothness s_z to make it look ok

def parameterize_Pi(logpi_spatial, Pi_temporal):
    """
    Parameterize Pi (precision matrix) with `logpi_spatial` 
    """
    Pi_spatial = jnp.diag(jnp.exp(logpi_spatial)) # multivariate Pi_spatial
    return jnp.kron(Pi_temporal, Pi_spatial)

alpha_all = initialization_dict['alpha'] * jnp.ones((N,)) # sample a different baseline alpha for every agent
eta0_all = initialization_dict['eta'] *jnp.ones((N,1,genmodel['ns_x'])) # fix the baseline eta for every sector, for every agent

def parameterize_f_params(f_params_pre):
    """
    Parameterize tilde eta (the attracting point of the flow function) with preparams, stop gradient on parameters of the A0 matrix
    """
    f_params = {
        'tilde_A': jnp.stack(genmodel['ndo_x'] * [parameterize_A0_no_coupling(lax.stop_gradient(f_params_pre['alpha']), genmodel['ns_x'])]), 
        'tilde_eta': jnp.concatenate((f_params_pre['eta0'], jnp.zeros((genmodel['ndo_x'] -1, genmodel['ns_x'] )))) # added lax.stop_gradient here to stop eta from getting updated
    }
    return f_params

preparams = {'logpiz_spatial': logpiz_spatial_all, 'logpiw_spatial': logpiw_spatial_all, 'f_params_pre': {'alpha': alpha_all, 'eta0': eta0_all}}
parameterization_mapping = {'logpiz_spatial': {'to_arg_name': 'Pi_z', 'fn': partial(parameterize_Pi, Pi_temporal=Pi_z_temporal)},
                            'logpiw_spatial': {'to_arg_name': 'Pi_w', 'fn': partial(parameterize_Pi, Pi_temporal=Pi_w_temporal)},
                            'f_params_pre': {'to_arg_name': 'f_params', 'fn': parameterize_f_params}
                            }                                    

initial_learnable_params = vmap(reparameterize, (0, None))(preparams, parameterization_mapping)
genmodel = merge(genmodel, initial_learnable_params)

mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)
def step_fn(carry, t):
    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)
    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['logpiz_spatial'], preparams['logpiw_spatial'], preparams['f_params_pre']['eta0'], F)
n_timesteps = len(genproc['t_axis'])
init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(n_timesteps))
position_history, velocity_history, mu_history, logpiz_hist, logpiw_hist, eta0_history = history[0], history[1], history[2], history[3], history[4], history[5]

h_hists = compute_sector_dists_over_time(position_history, velocity_history, genproc)

fig, axes = plt.subplots(2, 3, figsize=(12,8))
colors = cm.rainbow(np.linspace(0, 1, N))
from_end = n_timesteps

time_range = jnp.arange(n_timesteps-from_end, n_timesteps)
# time_range = jnp.arange(0, 20)

which_to_inspect = np.array(list(range(N)), dtype=int)
colors_subset = colors[which_to_inspect]
for i, c in zip(which_to_inspect, colors_subset):
    axes[0,0].plot(position_history[time_range,i,0], position_history[time_range,i,1], c=c)

which_particle = 0

axes[0,1].plot(time_range, mu_history[time_range,0,which_particle],label = 'VF 1 Inferred distance') 
axes[0,1].plot(time_range, h_hists[time_range,0,which_particle], label = 'VF 1 True Distance') 
axes[0,1].plot(time_range, eta0_history[time_range,which_particle,0,0], label = 'VF 1 Learned setpoint') # plot the learned setpoint in the first sector for the focal individual over time 
axes[0,1].legend()

axes[0,2].plot(time_range, mu_history[time_range,1,which_particle], label = 'VF 2 Inferred setpoint')
axes[0,2].plot(time_range, h_hists[time_range,1,which_particle], label = 'VF 2 True Distance')
axes[0,2].plot(time_range, eta0_history[time_range,which_particle,0,1], label = 'VF 2 Learned setpoint') # plot the learned setpoint in the second sector for the focal individual over time  
axes[0,2].legend()

axes[1,1].plot(time_range, logpiz_hist[time_range,which_particle,0], label = 'VF 1 Learned sensory precision') 
axes[1,1].plot(time_range, logpiw_hist[time_range,which_particle,0], label = 'VF 1 Learned process precision') 
axes[1,1].legend()

axes[1,2].plot(time_range, logpiz_hist[time_range,which_particle,1], label = 'VF 2 Learned sensory precision') 
axes[1,2].plot(time_range, logpiw_hist[time_range,which_particle,1], label = 'VF 2 Learned process precision') 
axes[1,2].legend()


plt.show()
