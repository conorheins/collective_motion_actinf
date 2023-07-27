"""
Script for running multiple perturbations in a sequence from a fixed point in time
"""

from jax import random, lax, vmap, jit, grad, jacfwd
from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

from utils import make_single_timestep_fn_nolearning, initialize_meta_params, get_default_inits
from genprocess import init_gen_process, get_observations_special
from genmodel import init_genmodel, compute_vfe_vectorized
from inference import run_inference

# set up some global variables (random key, T, dt, N, D)
key = random.PRNGKey(201)
T, dt, N, D = 100, 0.01, 50, 2
init_dict = get_default_inits(N, T, dt)

# initialize generative model, generative process, and meta parameters related to learning and inference
pos, vel, genproc, new_key = init_gen_process(key, init_dict)
n_timesteps = len(genproc['t_axis'])

genmodel = init_genmodel(init_dict)
meta_params = initialize_meta_params(infer_lr = 0.1, 
                                    nsteps_infer = 1, 
                                    action_lr = 0.1, 
                                    nsteps_action = 1, 
                                    normalize_v = True
                                    )

# initialize first beliefs using priors over hidden states
mu_init = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

# get single timestep function (no learning version)
single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)

# create a custom step function that will run the simulation as you want
def step_fn(carry, t):
    pos_past, vel_past, mu_past = carry
    pos, vel, mu, F = single_timestep(pos_past, vel_past, mu_past, t)
    return (pos, vel, mu), (pos, vel, mu, F)
init_state = (pos, vel, mu_init)
final_state, history = lax.scan(step_fn, init_state, jnp.arange(n_timesteps))
pos_hist, vel_hist, mu_hist = history[0], history[1], history[2]

def compute_group_VFE_angles(pos_t, angles_t, mu_t, t_idx):
    """ Function that computes the group free energy as a function of the positions, angles, and beliefs of a group of agents """

    # convert angles to velocity
    vel_t = jnp.concatenate([jnp.cos(angles_t).reshape(-1,1), jnp.sin(angles_t).reshape(-1,1)], axis = 1)

    # given positions and velocities, sample observations from generative process
    phi, all_dh_dr_self, empty_sectors_mask = get_observations_special(pos_t, vel_t, genproc, t_idx)

    # run hidden state inference where each agent minimizes free energy
    infer_res, mu_traj = run_inference(phi, mu_t, empty_sectors_mask, genmodel, **meta_params['inference_params'])
    mu_next, _ = infer_res

    # compute and return the resulting variational free energy, summed over individuals
    return compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel).sum()

dFall_dtheta = jit(grad(compute_group_VFE_angles, argnums = 1)) # create function that computes derivatives of group VFE with respect to heading direction (in angles) of all agents
angles_hist = vmap(jnp.arctan2)(vel_hist[:,:,1], vel_hist[:,:,0]) # convert velocity to angles
theta_grads = jnp.absolute(vmap(dFall_dtheta)(pos_hist, angles_hist, mu_hist, jnp.arange(n_timesteps)))

# def compute_group_VFE_velocities(pos_t, vel_t, mu_t, t_idx):
#     """ Function that computes the group free energy as a function of the positions, velocities, and beliefs of a group of agents """

#     # given positions and velocities, sample observations from generative process
#     phi, all_dh_dr_self, empty_sectors_mask = get_observations_special(pos_t, vel_t, genproc, t_idx)

#     # run hidden state inference where each agent minimizes free energy
#     infer_res, mu_traj = run_inference(phi, mu_t, empty_sectors_mask, genmodel, **meta_params['inference_params'])
#     mu_next, _ = infer_res

#     # compute and return the resulting variational free energy, summed over individuals
#     return compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel).sum()

# dFall_dvel= jit(grad(compute_group_VFE_velocities, argnums = 1)) # create function that computes derivatives of group VFE with respect to heading direction (in angles) of all agents
# dFall_dvel_evals = vmap(dFall_dvel)(pos_hist, vel_hist, mu_hist, jnp.arange(n_timesteps))
# v_gradient_norms = jnp.linalg.norm(dFall_dvel_evals,axis=2)

# metric_to_use = v_gradient_norms
metric_to_use = theta_grads

# differences in `metric_to_use` across the school'smembers
gradient_differences = jnp.amax(metric_to_use, axis = 1) - jnp.amin(metric_to_use, axis = 1)
perturb_start_t = int(jnp.argmax(gradient_differences)) # when that spread of gradients is largest, use that as the time to perturb

agent_id_max = int(jnp.argmax(metric_to_use[perturb_start_t])) 
agent_id_min = int(jnp.argmin(metric_to_use[perturb_start_t]))

# perturb_start_t, agent_id_max = jnp.where(metric_to_use == metric_to_use.max()) # find the time and agent index where the largest change in VFE is expected
# perturb_start_t, agent_id_max = int(perturb_start_t), int(agent_id_max)

# agent_id_min = jnp.where(metric_to_use[perturb_start_t,:] == metric_to_use[perturb_start_t,:].min())[0] # find the agent index within the same time slice, where the smallest change in VFE is expected
# agent_id_min = int(agent_id_min)

# n_perturbations, perturb_time, perturb_value, perturb_duration = 25, 3, -5.0, 5
n_perturbations, perturb_time, perturb_value, perturb_duration = 25, 3, jnp.pi/10.0, 5

keys = random.split(new_key, num=n_perturbations) # get a different random key for each perturbation

n_timesteps = len(jnp.arange(start=0, stop=perturb_time, step=dt))
angle_response_hist_maxdF = np.zeros((n_perturbations, n_timesteps, N))
F_response_hist_maxdF = np.zeros_like(angle_response_hist_maxdF)

angle_response_hist_mindF = np.zeros((n_perturbations, n_timesteps, N))
F_response_hist_mindF = np.zeros_like(angle_response_hist_mindF)

for (perturb_i, run_key) in enumerate(keys):

    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)

    """ First, run the maxDF version """
    # initialize generative model, generative process, and meta parameters related to learning and inference
    _, _, genproc, _ = init_gen_process(run_key, init_dict)

    """ First way of doing a perturbation: perturb the noise added to velocity observations  """
    # introduce a perturbation by adding incrementing the velocity observations of the chosen agent by a fixed value for a fixed duration of time 
    # noise_tensor = genproc['sensory_noise']
    # genproc['sensory_noise'] = noise_tensor.at[:perturb_duration,1,:,agent_id_max].set(noise_tensor[:perturb_duration,1,:,agent_id_max] + perturb_value)

    """ Second way of doing a perturbation: you use the actual gradient of dF/dvel or dF/dtheta to directly change the angle of the focal agent """
    new_angle = angles_hist[perturb_start_t,agent_id_max] + jnp.sign(metric_to_use[perturb_start_t, agent_id_max]) * perturb_value

    # wrap around if new_angle is greater than +pi or less than -pi
    if new_angle > jnp.pi:
        greater_than_pi = (new_angle - jnp.pi)
        new_angle = -jnp.pi + greater_than_pi
    elif new_angle < -jnp.pi:
        greater_than_pi = (new_angle - jnp.pi)
        new_angle = jnp.pi - greater_than_pi

    vel_init = vel_hist[perturb_start_t]
    vel_init = vel_init.at[agent_id_max,:].set(jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))

    # initialize the state to be the same for all perturbations, which is the chosen point in history `perturb_start_t` from the original simulation
    init_state = (pos_hist[perturb_start_t], vel_init, mu_hist[perturb_start_t])

    # @NOTE: We don't need to re-initialize the `genmodel` or `meta_params` dicts because in this example, that will be the same for all agents across the perturbations
    # However, if you do an example with learning you will need to re-initialize the `genmodel` for each perturbation
    # get single timestep function (no learning version)
    single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)
    # create a custom step function that will run the simulation as you want
    def step_fn(carry, t):
        pos_past, vel_past, mu_past = carry
        pos, vel, mu, F = single_timestep(pos_past, vel_past, mu_past, t)
        return (pos, vel, mu), (pos, vel, mu, F)
    _, history_perturb = lax.scan(step_fn, init_state, jnp.arange(n_timesteps))
    perturb_vel_i, perturb_F_i = history_perturb[1], history_perturb[3] # history of perturbed velocities and free energies
    angle_response_hist_maxdF[perturb_i] = jnp.absolute(vmap(jnp.arctan2)(perturb_vel_i[:,:,1], perturb_vel_i[:,:,0]))
    F_response_hist_maxdF[perturb_i] = perturb_F_i

    """ Next, run the minDF version """
    # initialize generative model, generative process, and meta parameters related to learning and inference
    _, _, genproc, _ = init_gen_process(run_key, init_dict)

    """ First way of doing a perturbation: perturb the noise added to velocity observations  """
    # introduce a perturbation by adding incrementing the velocity observations of the chosen agent by a fixed value for a fixed duration of time 
    # noise_tensor = genproc['sensory_noise']
    # genproc['sensory_noise'] = noise_tensor.at[:perturb_duration,1,:,agent_id_min].set(noise_tensor[:perturb_duration,1,:,agent_id_min] + perturb_value)

    """ Second way of doing a perturbation: you use the actual gradient of dF/dvel or dF/dtheta to directly change the angle of the focal agent """
    new_angle = angles_hist[perturb_start_t,agent_id_min] + jnp.sign(metric_to_use[perturb_start_t, agent_id_min]) * perturb_value

    # wrap around if new_angle is greater than +pi or less than -pi
    if new_angle > jnp.pi:
        greater_than_pi = (new_angle - jnp.pi)
        new_angle = -jnp.pi + greater_than_pi
    elif new_angle < -jnp.pi:
        greater_than_pi = (new_angle - jnp.pi)
        new_angle = jnp.pi - greater_than_pi

    vel_init = vel_hist[perturb_start_t]
    vel_init = vel_init.at[agent_id_min,:].set(jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))

    # initialize the state to be the same for all perturbations, which is the chosen point in history `perturb_start_t` from the original simulation
    init_state = (pos_hist[perturb_start_t], vel_init, mu_hist[perturb_start_t])

    # get single timestep function (no learning version)
    single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)
    # create a custom step function that will run the simulation as you want
    def step_fn(carry, t):
        pos_past, vel_past, mu_past = carry
        pos, vel, mu, F = single_timestep(pos_past, vel_past, mu_past, t)
        return (pos, vel, mu), (pos, vel, mu, F)
    _, history_perturb = lax.scan(step_fn, init_state, jnp.arange(n_timesteps))
    perturb_vel_i, perturb_F_i = history_perturb[1], history_perturb[3] # history of perturbed velocities and free energies
    angle_response_hist_mindF[perturb_i] = jnp.absolute(vmap(jnp.arctan2)(perturb_vel_i[:,:,1], perturb_vel_i[:,:,0]))
    F_response_hist_mindF[perturb_i] = perturb_F_i

# fig, axes = plt.subplots(2,3,figsize=(10,8))
# axes[0,0].plot(angle_response_hist_maxdF[0], c = 'b', lw = 0.25)
# axes[0,0].plot(angle_response_hist_maxdF[0].mean(-1), lw = 3.0)
# axes[0,1].plot(angle_response_hist_maxdF[1], c = 'b', lw = 0.25)
# axes[0,1].plot(angle_response_hist_maxdF[1].mean(-1), lw = 3.0)
# axes[0,2].plot(angle_response_hist_maxdF[2], c = 'b', lw = 0.25)
# axes[0,2].plot(angle_response_hist_maxdF[2].mean(-1), lw = 3.0)

# axes[1,0].plot(angle_response_hist_mindF[0], c = 'b', lw = 0.25)
# axes[1,0].plot(angle_response_hist_mindF[0].mean(-1), lw = 3.0)
# axes[1,1].plot(angle_response_hist_mindF[1], c = 'b', lw = 0.25)
# axes[1,1].plot(angle_response_hist_mindF[1].mean(-1), lw = 3.0)
# axes[1,2].plot(angle_response_hist_mindF[2], c = 'b', lw = 0.25)
# axes[1,2].plot(angle_response_hist_mindF[2].mean(-1), lw = 3.0)

fig, axes = plt.subplots(1,2,figsize=(10,8))

mean, error = angle_response_hist_maxdF.mean(-1).mean(axis=0), angle_response_hist_maxdF.mean(-1).std(axis=0)
axes[0].fill_between(jnp.arange(len(mean)), mean + error, mean - error, alpha = 0.3, color= 'b')
axes[0].plot(mean, color= 'b')

mean, error = angle_response_hist_mindF.mean(-1).mean(axis=0), angle_response_hist_mindF.mean(-1).std(axis=0)
axes[0].fill_between(jnp.arange(len(mean)), mean + error, mean - error, alpha = 0.3, color='r')
axes[0].plot(mean, color='r')

mean, error = F_response_hist_maxdF.mean(-1).mean(axis=0), F_response_hist_maxdF.mean(-1).std(axis=0)
axes[1].fill_between(jnp.arange(len(mean)), mean + error, mean - error, alpha = 0.3, color= 'b')
axes[1].plot(mean, color= 'b')

mean, error = F_response_hist_mindF.mean(-1).mean(axis=0), F_response_hist_mindF.mean(-1).std(axis=0)
axes[1].fill_between(jnp.arange(len(mean)), mean + error, mean - error, alpha = 0.3, color='r')
axes[1].plot(mean, color='r')

plt.show()


