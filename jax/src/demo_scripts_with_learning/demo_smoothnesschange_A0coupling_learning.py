from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np

from utils import make_single_timestep_fn, initialize_meta_params

from learning import make_dfdparams_fn
from genprocess import init_gen_process, generate_colored_noise
from genmodel import init_genmodel, parameterize_A0_with_coupling
from matplotlib import pyplot as plt

key = random.PRNGKey(4)

T, dt, N, D = 150, 0.01, 50, 2

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

# Block of code (needs to be wrapped into functions at some point) for parameterizing changes in autocorrelation of noise at different times
ns_phi, ndo_phi = initialization_dict['ns_phi'], initialization_dict['ndo_phi']
z_h, z_hprime = initialization_dict['z_h'], initialization_dict['z_hprime']
z_gp = jnp.array([z_h, z_hprime]).reshape(1, ndo_phi, 1, 1)
noise_colors = [0.1, 1.0] # coloredness of noise at two different times
smoothness_change_t = int((len(genproc['t_axis'])/2)) # time at which autocorrelation changes from `noise_colors[0]` to `noise_colors[1]`
n_remaining = len(genproc['t_axis']) - smoothness_change_t
first_half_noise = z_gp * generate_colored_noise(beta=noise_colors[0], N=N, n_dim=ns_phi, n_timesteps=smoothness_change_t)
second_half_noise = z_gp * generate_colored_noise(beta=noise_colors[1], N=N, n_dim=ns_phi, n_timesteps=n_remaining)
genproc['sensory_noise'] = jnp.vstack([first_half_noise, second_half_noise])


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
_, alpha_key = random.split(new_key)

alpha_all = random.uniform(alpha_key, minval = 0.25, maxval = 0.75, shape = (N,)) # sample a different baseline alpha for every agent
alpha_beta_all = jnp.hstack( ( alpha_all[...,None], 0.5 * alpha_all[...,None]))   # beta is simply half of alpha for every agent

eta0_all = initialization_dict['eta'] * jnp.ones((N,1,genmodel['ns_x']))

def parameterize_f_params(f_params_pre):
    """
    Version where you're learning the coupling coefficients of the A matrix
    """
    f_params = {
        'tilde_A': jnp.stack(genmodel['ndo_x'] * [parameterize_A0_with_coupling(f_params_pre['alpha_beta'], genmodel['ns_x'])]), 
        'tilde_eta': jnp.concatenate((lax.stop_gradient(f_params_pre['eta0']), jnp.zeros((genmodel['ndo_x']-1, genmodel['ns_x'])))) # added lax.stop_gradient here to stop eta from getting updated
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

    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['f_params_pre']['alpha_beta'], F)

init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))

position_history = np.array(history[0])

fig, axes = plt.subplots(1, 3, figsize=(10,8))
for i in range(N):
    # axes[0].plot(position_history[:,i,0], position_history[:,i,1])
    axes[0].plot(position_history[:smoothness_change_t,i,0], position_history[:smoothness_change_t,i,1], c = 'b')
    axes[0].plot(position_history[smoothness_change_t:,i,0], position_history[smoothness_change_t:,i,1], c = 'r')

alpha_hist = history[3][:,:,0]
beta_hist = history[3][:,:,1]

# axes[1].plot(jnp.arange(0, len(genproc['t_axis'])),alpha_hist)
# axes[2].plot(jnp.arange(0, len(genproc['t_axis'])),beta_hist)

good_fish_ids = []
for n_idx in range(N):

    if jnp.all(alpha_hist[smoothness_change_t:,n_idx] < 10.0):
        good_fish_ids.append(n_idx)

good_fish_ids = jnp.array(good_fish_ids)

burn_in_time = 1000
axes[1].plot(jnp.arange(start = burn_in_time, stop = smoothness_change_t),alpha_hist[burn_in_time:smoothness_change_t,good_fish_ids], c = 'b', label = 'before smoothness change')
axes[1].plot(jnp.arange(start = smoothness_change_t, stop = len(genproc['t_axis'])), alpha_hist[smoothness_change_t:,good_fish_ids], c = 'r', label = 'after smoothness change')

axes[2].plot(jnp.arange(start = burn_in_time, stop = smoothness_change_t),beta_hist[burn_in_time:smoothness_change_t,good_fish_ids], c = 'b', label = 'before smoothness change')
axes[2].plot(jnp.arange(start = smoothness_change_t, stop = len(genproc['t_axis'])), beta_hist[smoothness_change_t:,good_fish_ids], c = 'r', label = 'after smoothness change')
plt.show()
