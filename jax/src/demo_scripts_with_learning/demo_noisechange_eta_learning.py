from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np

from utils import make_single_timestep_fn, initialize_meta_params

from learning import make_dfdparams_fn
from genprocess import init_gen_process, change_noise_variance
from genmodel import init_genmodel, parameterize_A0_no_coupling
from matplotlib import pyplot as plt

key = random.PRNGKey(4)

T, dt, N, D = 500, 0.01, 25, 2

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
_, eta_key = random.split(new_key)

alpha_all = initialization_dict['alpha'] * jnp.ones((N,)) # sample a different baseline alpha for every agent

# eta0_all = random.uniform(eta_key, minval = 0.95, maxval = 1.05, shape = (N,1,1)) # sample a different baseline eta for every agent, that is duplicated across sectors
# eta0_all = eta0_all * jnp.ones((1,1,genmodel['ns_x']))

eta0_all = random.uniform(eta_key, minval = 0.95, maxval = 1.05, shape = (N,1,genmodel['ns_x'])) # sample a different baseline eta for every sector, for every agent

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

init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))

position_history = history[0]

fig, axes = plt.subplots(1, 2, figsize=(10,8))
for i in range(N):
    # axes[0].plot(position_history[:,i,0], position_history[:,i,1])
    axes[0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    axes[0].plot(position_history[noise_change_t:,i,0], position_history[noise_change_t:,i,1], c = 'r')

eta0_history = history[3]

good_fish_ids = []
for n_idx in range(N):
    if jnp.all(eta0_history[noise_change_t:,n_idx,:,:] < 20.0):
        good_fish_ids.append(n_idx)
good_fish_ids = jnp.array(good_fish_ids)

_, sampling_key = random.split(eta_key)

# plot beliefs of all fish about fixed point of some particular sector over time
burn_in_time = 1000

# sector_idx = random.choice(sampling_key, jnp.arange(genmodel['ns_x']))
sector_idx = 3
axes[1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),eta0_history[burn_in_time:noise_change_t,good_fish_ids, 0, sector_idx], c = 'b', label = 'before noise change')
axes[1].plot(jnp.arange(start = noise_change_t, stop = len(genproc['t_axis'])), eta0_history[noise_change_t:, good_fish_ids, 0, sector_idx], c = 'r', label = 'after noise change')

# particle_idx = random.choice(sampling_key, jnp.arange(N))
# axes[1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),eta0_history[burn_in_time:noise_change_t,particle_idx, 0, :], c = 'b', label = 'before noise change')
# axes[1].plot(jnp.arange(start = noise_change_t, stop = len(genproc['t_axis'])), eta0_history[noise_change_t:, particle_idx, 0, :], c = 'r', label = 'after noise change')

plt.show()
