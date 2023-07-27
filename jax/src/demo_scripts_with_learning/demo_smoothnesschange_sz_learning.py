from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from funcy import merge

from utils import make_single_timestep_fn, initialize_meta_params

from learning import make_dfdparams_fn, reparameterize
from genprocess import init_gen_process, generate_colored_noise
from genmodel import init_genmodel
from matplotlib import pyplot as plt

key = random.PRNGKey(2)

T, dt, N, D = 100, 0.01, 50, 2

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
noise_colors = [0.01, 5.0] # coloredness of noise at two different times
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
_, smoothness_key = random.split(new_key)

s_z_all = random.uniform(smoothness_key, minval = 0.95, maxval = 1.05, shape = (N,)) # sample a different sensory smoothness for every agent

Pi_z_spatial = initialization_dict['pi_z_spatial'] * jnp.eye(genmodel['ns_phi'])
def parameterize_Pi_2do(s_z):
    """
    Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
    """
    Pi_z_temporal = jnp.diag(jnp.array([1.0, 2 * s_z**2]))
    # Pi_z_temporal, _ = create_temporal_precisions(genmodel['ndo_phi'], s_z)
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

position_history = history[0]

fig, axes = plt.subplots(1, 2, figsize=(10,8))
for i in range(N):
    # axes[0].plot(position_history[:,i,0], position_history[:,i,1])
    axes[0].plot(position_history[:smoothness_change_t,i,0], position_history[:smoothness_change_t,i,1], c = 'b')
    axes[0].plot(position_history[smoothness_change_t:,i,0], position_history[smoothness_change_t:,i,1], c = 'r')

sz_history = history[3]
# axes[1].plot(jnp.arange(0, len(genproc['t_axis'])), sz_history) # plot beliefs about first sector eta for all individuals over time

burn_in_time = 1000
axes[1].plot(jnp.arange(start = burn_in_time, stop = smoothness_change_t),sz_history[burn_in_time:smoothness_change_t], c = 'b', label = 'before smoothness change')
axes[1].plot(jnp.arange(start = smoothness_change_t, stop = len(genproc['t_axis'])), sz_history[smoothness_change_t:], c = 'r', label = 'after smoothness change')
plt.show()
