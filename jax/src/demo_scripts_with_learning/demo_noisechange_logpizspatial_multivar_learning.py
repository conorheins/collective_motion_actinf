from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np

from utils import make_single_timestep_fn, initialize_meta_params

from learning import make_dfdparams_fn
from genprocess import init_gen_process, change_noise_variance
from genmodel import init_genmodel, create_temporal_precisions
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
                        'z_h': 0.5,      # variance of additive observation noise on first order hidden states
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
noise_change_scalar = 2.0
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
_, piz_key = random.split(new_key)

logpi_z_spatial = 0.5 * random.normal(piz_key, shape = (N,genmodel['ns_phi'])) # sample a different sensory spatial precision for every sector, for every agent

Pi_z_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_phi'], smoothness=initialization_dict['s_z'])

def parameterize_Pi_z(logpi_z_spatial):
    """
    Parameterize Pi_z (sensory preicison matrix) with `pi_z_spatial` where a different sensory precision assigned to each sector
    """
    return jnp.kron(Pi_z_temporal, jnp.diag(jnp.exp(logpi_z_spatial)))

preparams = {'logpi_z_spatial': logpi_z_spatial}
parameterization_mapping = {'logpi_z_spatial': {'to_arg_name': 'Pi_z', 'fn': parameterize_Pi_z}}
genmodel['Pi_z'] = vmap(parameterize_Pi_z)(preparams['logpi_z_spatial'])

mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)

def step_fn(carry, t):

    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)

    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['logpi_z_spatial'], F)

init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))

position_history = history[0]

fig, axes = plt.subplots(1, 3, figsize=(10,10))
for i in range(N):
    axes[0].plot(position_history[:,i,0], position_history[:,i,1])
    axes[0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    axes[0].plot(position_history[noise_change_t:,i,0], position_history[noise_change_t:,i,1], c = 'r')

logpizspatial_history = history[3]

good_fish_ids = []
for n_idx in range(N):

    if jnp.all(logpizspatial_history[noise_change_t:,n_idx,:] < 5.0):
        good_fish_ids.append(n_idx)
    
good_fish_ids = jnp.array(good_fish_ids)
variance_history = 1. / jnp.exp(logpizspatial_history[:,good_fish_ids,:])

burn_in_time = 1000
_, sampling_key = random.split(piz_key)

particle_idx = random.choice(sampling_key, jnp.arange(variance_history.shape[1]))
axes[1].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),variance_history[burn_in_time:noise_change_t, particle_idx, :], c = 'b', label = 'before noise change') # plot beliefs about all sector's pi_z for one individual over time
axes[1].plot(jnp.arange(start = noise_change_t, stop = len(genproc['t_axis'])), variance_history[noise_change_t:, particle_idx, :], c = 'r', label = 'after noise change') # plot beliefs about all sector's pi_z for one individual over time

_, sampling_key = random.split(sampling_key)
sector_idx = random.choice(sampling_key, jnp.arange(genmodel['ns_phi']))
axes[2].plot(jnp.arange(start = burn_in_time, stop = noise_change_t),variance_history[burn_in_time:noise_change_t,:,sector_idx]) # plot beliefs about one sector's pi_z for all individuals over time
axes[2].plot(jnp.arange(start = noise_change_t, stop = len(genproc['t_axis'])), variance_history[noise_change_t:,:,sector_idx]) # plot beliefs about one sector's pi_z for all individuals over time

plt.show()
