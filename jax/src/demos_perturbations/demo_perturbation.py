import jax
from jax import random, lax, vmap, jit, grad
from jax import numpy as jnp
import numpy as np
import pickle
from funcy import merge

from utils import make_single_timestep_fn, make_single_timestep_fn_nolearning, initialize_meta_params, get_default_inits
from learning import reparameterize, make_dfdparams_fn
from genprocess import init_gen_process, get_observations_special, compute_turning_magnitudes
from genmodel import init_genmodel, compute_vfe_vectorized
from inference import run_inference

T, dt, N, D = 75, 0.01, 50, 2 # for small computer

n_timesteps = len(jnp.arange(start=0, stop=T, step=dt))
n_perturbations, perturb_time, perturb_value, perturb_duration = 2, 10, -5., 5 # for small computer

init_key = random.PRNGKey(1)
init_dict = get_default_inits(N, T, dt)

# initialize generative model, generative process, and meta parameters related to learning and inference
pos, vel, genproc, new_key = init_gen_process(init_key, init_dict)
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

def compute_group_VFE_velocities(pos_t, vel_t, mu_t, t_idx):
    """ Function that computes the group free energy as a function of the positions, velocities, and beliefs of a group of agents """
    phi, all_dh_dr_self, empty_sectors_mask = get_observations_special(pos_t, vel_t, genproc, t_idx)
    infer_res, mu_traj = run_inference(phi, mu_t, empty_sectors_mask, genmodel, **meta_params['inference_params'])
    mu_next, _ = infer_res
    return compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel).sum()

dFall_dvel= jit(grad(compute_group_VFE_velocities, argnums = 1)) # create function that computes derivatives of group VFE with respect to heading direction (in angles) of all agents
dFall_dvel_eval = dFall_dvel(pos_hist[-1], vel_hist[-1], mu_hist[-1], n_timesteps-1)
v_gradient_norm = jnp.linalg.norm(dFall_dvel_eval,axis=1) # take the norm of each of these gradient vectors, one norm computed per agent
agent_id = jnp.argmax(v_gradient_norm) # find the agent for whom this norm is largest

perturb_keys = random.split(new_key, num=n_perturbations) # get a different random key for each perturbation
n_timesteps_perturb = len(jnp.arange(start=0, stop=perturb_time, step=dt))

def run_perturbation(perturb_key):
    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)

    # initialize generative model, generative process, and meta parameters related to learning and inference
    _, _, genproc, _ = init_gen_process(perturb_key, init_dict)
    genmodel = init_genmodel(init_dict)

    # introduce a perturbation by incrementing the velocity observations of the chosen agent by a fixed value for a fixed duration of time 
    noise_tensor = genproc['sensory_noise']
    genproc['sensory_noise'] = noise_tensor.at[:perturb_duration,1,:,agent_id].set(noise_tensor[:perturb_duration,1,:,agent_id] + perturb_value)

    # initialize the state to be the same for all perturbations, which is the last point in history of the original simulation
    init_state = (pos_hist[-1], vel_hist[-1], mu_hist[-1])

    single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)
    # create a custom step function that will run the simulation as you want
    def step_fn(carry, t):
        pos_past, vel_past, mu_past = carry
        pos, vel, mu, F = single_timestep(pos_past, vel_past, mu_past, t)
        return (pos, vel, mu), (vel, F)
    _, history_perturb_nolearn = lax.scan(step_fn, init_state, jnp.arange(n_timesteps_perturb))
    perturb_vel_i, F_responses_nolearn = history_perturb_nolearn[0], history_perturb_nolearn[1] # history of perturbed velocities and free energies
    reference_velocity = init_state[1]
    turning_angles_raw = compute_turning_magnitudes(reference_velocity, perturb_vel_i)

    return turning_angles_raw

turning_angles_raw = vmap(run_perturbation)(perturb_keys)

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1,2, figsize=(10,8))
axes[0].plot(turning_angles_raw[0])
axes[1].plot(turning_angles_raw[1])

plt.show()



