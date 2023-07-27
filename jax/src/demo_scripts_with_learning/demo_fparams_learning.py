from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np

from utils import make_single_timestep_fn, initialize_meta_params

from learning import make_dfdparams_fn
from genprocess import init_gen_process
from genmodel import init_genmodel, parameterize_A0_with_coupling
from matplotlib import pyplot as plt

key = random.PRNGKey(4)

T, dt, N, D = 50, 0.01, 50, 2

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
                        's_z': 1.0,          # the assumed smoothness (temporal autocorrelation) of sensory fluctuations
                        's_w': 1.0           # the assumed smoothness (temporal autocorrelation) of process fluctuations
}
pos, vel, genproc, new_key = init_gen_process(key, initialization_dict)

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
alpha_key, eta_key = random.split(new_key)

alpha_all = random.uniform(alpha_key, minval = 0.45, maxval = 0.55, shape = (N,)) # sample a different baseline alpha for every agent
alpha_beta_all = jnp.hstack( ( alpha_all[...,None], 0.5 * alpha_all[...,None]))   # beta is simply 0.5 * alpha for every agent

eta0_all = random.uniform(eta_key, minval = 0.95, maxval = 1.05, shape = (N,1,1)) # sample a different baseline eta for every agent
eta0_all = eta0_all * jnp.ones((1,1,genmodel['ns_x']))

def parameterize_f_params(f_params_pre):
    """
    Version where you're learning the coupling coefficients of the A matrix
    """
    f_params = {
        'tilde_A': jnp.stack(genmodel['ndo_x'] * [parameterize_A0_with_coupling(f_params_pre['alpha_beta'], genmodel['ns_x'])]), 
        'tilde_eta': jnp.concatenate((f_params_pre['eta0'], jnp.zeros((genmodel['ndo_x'] -1, genmodel['ns_x'] )))) # added lax.stop_gradient here to stop eta from getting updated
    }
    return f_params

preparams = {'f_params_pre': {'alpha_beta': alpha_beta_all, 'eta0': eta0_all}}

parameterization_mapping = {'f_params_pre': {'to_arg_name': 'f_params', 'fn': parameterize_f_params} }

genmodel['f_params'] = vmap(parameterize_f_params)(preparams['f_params_pre'])

mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)

def step_fn(carry, t):

    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)

    # return (pos, vel, mu, preparams), (pos, vel, mu, preparams['f_params_pre']['alpha'], preparams['f_params_pre']['eta0'], F)
    # return (pos, vel, mu, preparams), (pos, vel, mu, preparams['f_params_pre']['A0'], preparams['f_params_pre']['eta0'], F)
    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['f_params_pre']['alpha_beta'], preparams['f_params_pre']['eta0'], F)

init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))

# dist_matrix_hist = geo.compute_dist_matrices_over_time(history[0])
# print(dist_matrix_hist.shape)

# vis_neighbours_hist, distance_matrix_hist, _ = vmap(geo.compute_visual_neighbours, (0, 0, None, None, None))(history[0], history[1], genproc['R_starts'], genproc['R_ends'], genproc['dist_thr'])
# print(vis_neighbours_hist.shape)
# print(distance_matrix_hist.shape)

position_history = np.array(history[0])

fig, axes = plt.subplots(1, 2, figsize=(10,8))
for i in range(N):
    # axes[0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    # axes[0].plot(position_history[noise_change_t:,i,0], position_history[noise_change_t:,i,1], c = 'r')
    axes[0].plot(position_history[:,i,0], position_history[:,i,1])

# # axes[1].plot(history[2][:,:4,1])
# # # axes[1].plot()
# # # axes[1].plot(history[-1].mean(axis=1))
# # print(history[4].shape)
# # axes[1].plot(jnp.arange(0, len(t_axis)), history[4][:,:,0,0]) # plot beliefs about first sector eta for all individuals over time
# # axes[1].plot(jnp.arange(0, len(t_axis)), history[4][:,3,0,:]) # plot beliefs about all sector etas for one individual over time

# axes[1].plot(jnp.arange(0, len(t_axis)), history[3][:,:,0]) # this is the alpha parameters
# # axes[1].plot(jnp.arange(0, len(t_axis)), history[3][:,:,1]) # this is the beta parameters

# # axes[1].plot(jnp.arange(start = 0, stop = noise_change_t), history[3][:noise_change_t].mean(-1), c = 'b', label = 'before noise change')
# # axes[1].plot(jnp.arange(start = noise_change_t, stop = len(t_axis)), history[3][noise_change_t:].mean(-1), c = 'r', label = 'after noise change')

plt.show()
