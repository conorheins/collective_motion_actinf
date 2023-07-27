from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np

import genprocess.geometry as geo 
from inference import run_inference_as_Pi_z
from action import infer_actions
from learning import update_sz
from genmodel import compute_vfe_vectorized

from genprocess import sensory_samples_multi_order, identity_transform, grad_identity, get_observations, generate_colored_noise
from genmodel import g_tilde_linear, grad_g_tilde_linear, f_tilde_linear, grad_f_tilde_linear, create_temporal_precisions_jax
from matplotlib import pyplot as plt

# initialize stuff

key = random.PRNGKey(4)

T, dt, N, D = 50, 0.01, 45, 2
t_axis = jnp.arange(start=0, stop=T, step=dt)

pos_key, vel_key, smooth_key, noisekey1, noisekey2 = random.split(key, 5)

pos = random.uniform(pos_key, minval=-1.0, maxval = 1.0, shape = (N, D))
vel = geo.normalize_array(random.uniform(vel_key, minval=-1.0, maxval = 1.0, shape = (N, D)), axis = 1)
s_z = random.uniform(smooth_key, minval = 0.95, maxval = 1.05, shape = (N,)) # sample a different sensory smoothness for every agent
# s_z = random.uniform(smooth_key, minval = 2.45, maxval = 2.55, shape = (N,)) # sample a different sensory smoothness for every agent

sector_angles = [120., 60., 0., 360. - 60., 360. - 120.]

ns_x = len(sector_angles)-1
ns_phi = ns_x

ndo_x, ndo_phi = 3,2

genproc = {}
genproc['R_starts'], genproc['R_ends'] = geo.compute_rotation_matrices(sector_angles, reverse_flag=True)
genproc['dist_thr'] = 5.0

z_h, z_hprime = 0.1, 0.1
z_gp = jnp.array([z_h, z_hprime]).reshape(1, ndo_phi, 1, 1)


hprime_colors = [1.0, 0.1]
noise_change_t = int((len(t_axis)/2))
first_half_noise = z_gp * generate_colored_noise(beta=hprime_colors[0], N=N, n_dim=ns_phi, n_timesteps=noise_change_t)

n_remaining = len(t_axis) - noise_change_t
second_half_noise = z_gp * generate_colored_noise(beta=hprime_colors[1], N=N, n_dim=ns_phi, n_timesteps=n_remaining)

noise_tensor = jnp.vstack([first_half_noise, second_half_noise])


# use this if you want to change the noise magnitude halfway through

# noise_tensor = z_gp * random.normal(noisekey1, shape = (len(t_axis), ndo_phi, ns_phi, N))
# noise_change_t = int((len(t_axis)/2))

# # noise_change_scalar = 0.01
# # noise_tensor = noise_tensor.at[noise_change_t:,:,:,:].set(noise_change_scalar * noise_tensor[noise_change_t:,:,:,:]) # increase variance of noise by 3 halfway through

genproc['sensory_noise'] = noise_tensor

genproc['sensory_transform'] = identity_transform
genproc['grad_sensory_transform'] = grad_identity

z_action = 0.01
genproc['action_noise'] = z_action * random.normal(noisekey1, shape = (len(t_axis), N, 2))

genmodel = {}

alpha = 0.5
eta = 1.
pi_z_spatial = 1.0
pi_w_spatial = 1.0

genmodel['g'] = g_tilde_linear
genmodel['grad_g'] = grad_g_tilde_linear
genmodel['g_params'] = {}
g0_all = jnp.stack(N * [jnp.eye(ns_phi)], axis = 0)
genmodel['g_params']['tilde_g'] = jnp.stack(ndo_phi * [g0_all], axis = 1)
genmodel['pi_z_spatial'] = pi_z_spatial
genmodel['pi_w_spatial'] = pi_w_spatial

def generate_precision_matrix(smoothness):
    spatial = pi_z_spatial * jnp.eye(ns_phi)
    temporal = create_temporal_precisions_jax(ndo_phi, smoothness)[0]
    return jnp.kron(temporal, spatial)

genmodel['Pi_z'] = vmap(generate_precision_matrix)(s_z)

genmodel['f'] = f_tilde_linear
genmodel['grad_f'] = grad_f_tilde_linear
genmodel['f_params'] = {}
A0_all = jnp.stack(N * [alpha * jnp.eye(ns_x)], axis = 0)
eta0_all = eta * jnp.ones((N,1,ns_x))
genmodel['f_params']['tilde_A'] = jnp.stack(ndo_x * [A0_all], axis = 1)
genmodel['f_params']['tilde_eta'] = jnp.concatenate((eta0_all, jnp.zeros((N, ndo_x-1, ns_x))), axis = 1)

genmodel['Pi_w'] =  jnp.stack(N * [pi_w_spatial * jnp.eye(ns_x*ndo_x)], axis = 0)

genmodel['D_shift'] = jnp.diag(jnp.ones((ns_x*ndo_x- ns_x)), k = ns_x)
genmodel['D_T'] = genmodel['D_shift'].T

genmodel['ns_phi'], genmodel['ndo_phi'], genmodel['ns_x'], genmodel['ndo_x'] = ns_phi, ndo_phi, ns_x, ndo_x
genproc['dt'] = dt

inference_params = {'k_mu': 0.1, 
                    'num_steps': 1
                }

action_params = {'k_alpha': 0.1,
                  'num_steps': 1,
                  'normalize_v': True
                }

learning_params = {'k_param': 0.01,
                   'num_steps': 1
                }

mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, ndo_x*ns_x).T


def parameterize_pi_z_dumb(s_z):
    spatial = pi_z_spatial * jnp.eye(ns_phi)
    temporal = jnp.diag(jnp.array([1.0, 2 * s_z**2]))
    return jnp.kron(temporal, spatial)

def single_timestep(pos, vel, mu, s_z, t_idx):

    # sample observations from generative process
    phi, all_dh_dr_self, empty_sectors_mask = get_observations(pos, vel, genproc, t_idx)

    # run hidden state inference 
    Pi_z = vmap(parameterize_pi_z_dumb)(s_z)
    infer_res, mu_traj = run_inference_as_Pi_z(phi, mu, Pi_z, empty_sectors_mask, genmodel, **inference_params)
    mu_next, epsilon_z = infer_res

    # compute variational free energy 
    F = compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel)

    # use results of inference to update actions
    vel_next = infer_actions(vel, epsilon_z, genmodel, all_dh_dr_self, **action_params)

    # use actions to update generative process
    pos_next = geo.advance_positions(pos, vel_next, genproc['action_noise'][t_idx], dt = genproc['dt'])

    # update generative model parameters
    s_z_next = update_sz(s_z, mu, phi, empty_sectors_mask, genmodel, **learning_params)

    return pos_next, vel_next, mu_next, s_z_next, F

def step_fn(carry, t):

    pos_past, vel_past, mu_past, s_z_past = carry
    pos, vel, mu, s_z, F = single_timestep(pos_past, vel_past, mu_past,s_z_past, t)

    return (pos, vel, mu, s_z), (pos, vel, mu, s_z, F)

init_state = (pos, vel, mu, s_z)
out, history = lax.scan(step_fn, init_state, jnp.arange(len(t_axis)))

print(history[2].shape)
from matplotlib import pyplot as plt

position_history = np.array(history[0])

fig, axes = plt.subplots(1, 2, figsize=(10,8))
for i in range(N):
    axes[0].plot(position_history[:noise_change_t,i,0], position_history[:noise_change_t,i,1], c = 'b')
    axes[0].plot(position_history[noise_change_t:,i,0], position_history[noise_change_t:,i,1], c = 'r')

# axes[1].plot(history[2][:,:4,1])
# axes[1].plot()
# axes[1].plot(history[-1].mean(axis=1))
axes[1].plot(jnp.arange(start = 0, stop = noise_change_t), history[3][:noise_change_t].mean(-1), c = 'b', label = 'before noise change')
axes[1].plot(jnp.arange(start = noise_change_t, stop = len(t_axis)), history[3][noise_change_t:].mean(-1), c = 'r', label = 'after noise change')

plt.show()
