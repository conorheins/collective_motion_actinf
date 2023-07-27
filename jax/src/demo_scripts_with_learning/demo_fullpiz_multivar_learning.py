from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np

from utils import make_single_timestep_fn, initialize_meta_params

from learning import make_dfdparams_fn
from genprocess import init_gen_process
from genmodel import init_genmodel
from matplotlib import pyplot as plt

key = random.PRNGKey(4)

T, dt, N, D = 50, 0.01, 30, 2

initialization_dict = {'N': N,
                        'posvel_init': {'pos_x_bounds': [-5., 5.],
                                        'pos_y_bounds': [-5., 5.],
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
s_z_key, piz_key = random.split(new_key)

s_z_all = random.uniform(s_z_key, minval = 0.95, maxval = 1.05, shape = (N,)) # sample a different sensory temporal precision for every agent
pi_z_spatial_all = random.uniform(piz_key, minval = 0.95, maxval = 1.05, shape = (N,genmodel['ns_phi'])) # sample a different sensory precision for every sector, for every agent

def parameterize_Pi_z(Pi_z_preparams):
    """
    Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
    """
    pi_z_spatial, s_z = Pi_z_preparams['pi_z_spatial'], Pi_z_preparams['s_z'] # pull out parameters
    
    Pi_z_spatial = jnp.diag(pi_z_spatial) # multivariate pi_z_spatial
    Pi_z_temporal = jnp.diag(jnp.array([1.0, 2 * s_z**2]))
    return jnp.kron(Pi_z_temporal, Pi_z_spatial)

preparams = {'Pi_z_preparams': {'pi_z_spatial': pi_z_spatial_all, 's_z': s_z_all}}
parameterization_mapping = {'Pi_z_preparams': {'to_arg_name': 'Pi_z', 'fn': parameterize_Pi_z}}
genmodel['Pi_z'] = vmap(parameterize_Pi_z)(preparams['Pi_z_preparams'])

mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)

def step_fn(carry, t):

    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)

    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['Pi_z_preparams']['pi_z_spatial'], preparams['Pi_z_preparams']['s_z'], F)

init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))

position_history = np.array(history[0])

fig, axes = plt.subplots(1, 4, figsize=(10,10))
for i in range(N):
    axes[0].plot(position_history[:,i,0], position_history[:,i,1])

_, sampling_key = random.split(piz_key)
particle_idx = random.choice(sampling_key, jnp.arange(N))

pizspatial_history = np.array(history[3])
axes[1].plot(jnp.arange(0, len(genproc['t_axis'])), pizspatial_history[:,particle_idx,:]) # plot beliefs about all sector pi_z for one individual over time

_, sampling_key = random.split(sampling_key)
sector_idx = random.choice(sampling_key, jnp.arange(genmodel['ns_phi']))
axes[2].plot(jnp.arange(0, len(genproc['t_axis'])), pizspatial_history[:,:,sector_idx]) # plot beliefs about a particular sector's pi_z for all individuals over time

sz_history = np.array(history[4])
axes[3].plot(jnp.arange(0, len(genproc['t_axis'])), sz_history) # plot beliefs about s_z for all individuals over time

plt.show()
