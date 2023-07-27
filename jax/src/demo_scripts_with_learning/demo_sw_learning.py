from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np

from utils import make_single_timestep_fn, initialize_meta_params

from learning import make_dfdparams_fn
from genprocess import init_gen_process
from genmodel import init_genmodel
from matplotlib import pyplot as plt

key = random.PRNGKey(1)

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
                        'eta': 1.0,       # the fixed point of the flow function
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
_, smoothness_key = random.split(new_key)

s_w_all = random.uniform(smoothness_key, minval = 0.95, maxval = 1.05, shape = (N,)) # sample a different sensory smoothness for every agent

Pi_w_spatial = initialization_dict['pi_w_spatial'] * jnp.eye(genmodel['ns_x'])
def parameterize_Pi_3do(s_w):
    """
    Parameterize Pi_w (process precision matrix) with `s_w` under assumption of three orders of motion
    """
    Pi_w_temporal = jnp.diag(jnp.array([1.5, 2*s_w**2, 2*s_w**4])) + s_w**2* jnp.eye(genmodel['ndo_x'], k = 1) + s_w**2* jnp.eye(genmodel['ndo_x'], k = -1)
    # Pi_w_temporal, _ = create_temporal_precisions(genmodel['ndo_x'], s_w)
    return jnp.kron(Pi_w_temporal, Pi_w_spatial)

preparams = {'s_w': s_w_all}
parameterization_mapping = {'s_w': {'to_arg_name': 'Pi_w', 'fn': parameterize_Pi_3do}}
genmodel['Pi_w'] = vmap(parameterize_Pi_3do)(preparams['s_w'])

mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)

def step_fn(carry, t):

    pos_past, vel_past, mu_past, preparams_past = carry
    pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)

    return (pos, vel, mu, preparams), (pos, vel, mu, preparams['s_w'], F)

init_state = (pos, vel, mu, preparams)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))

position_history = np.array(history[0])

fig, axes = plt.subplots(1, 2, figsize=(10,8))
for i in range(N):
    axes[0].plot(position_history[:,i,0], position_history[:,i,1])

sw_history = np.array(history[3])
axes[1].plot(jnp.arange(0, len(genproc['t_axis'])), sw_history) # plot beliefs about process fluctuation smoothness for all individuals over time

plt.show()
