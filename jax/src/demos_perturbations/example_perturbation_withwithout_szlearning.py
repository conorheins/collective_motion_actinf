"""
Script for running multiple perturbations in a sequence from a fixed point in time, while varying the ability to learn vs. not learn parameters
Will start with s_z and adapt if needed
"""

import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=6' if you're using multi-processing try this
import argparse
import jax
from jax import random, lax, vmap, jit, grad
# from jax import pmap
from jax import numpy as jnp
import numpy as np
import pickle
from funcy import merge

from utils import make_single_timestep_fn, make_single_timestep_fn_nolearning, initialize_meta_params, get_default_inits
from learning import reparameterize, make_dfdparams_fn
from genprocess import init_gen_process, get_observations_special, compute_turning_magnitudes
from genmodel import init_genmodel, compute_vfe_vectorized
from inference import run_inference

from matplotlib import pyplot as plt

# cpus = jax.devices("cpu")
# gpus = jax.devices("gpu")

def parameterize_Pi_2do(s_z):
    """
    Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
    """
    Pi_z_temporal = jnp.diag(jnp.array([1.0, 2 * s_z**2]))
    return jnp.kron(Pi_z_temporal, jnp.eye(4))

def run(init_key_num = 1, # number to initialize the jax random seed
        N = 50, # number of individuals
        T = 75, # number of timesteps to run the original simulation
        perturb_length = 10, #  duration of time (in seconds) of simulation recorded following the perturbation
        perturb_value = -5., # scalar value of the perturbation
        perturb_duration = 10, # number of timesteps that the change to the observations lasts (the perturbation intervention itself)
        n_pre = 1000 # number of timesteps to show trajectories pre-perturbation
        ):

    init_key = random.PRNGKey(init_key_num)

    dt, D = 0.01, 2 
    
    init_dict = get_default_inits(N, T, dt)

    # initialize generative model, generative process, and meta parameters related to learning and inference
    pos, vel, genproc, new_key = init_gen_process(init_key, init_dict)
    n_timesteps = len(genproc['t_axis'])
    
    genmodel = init_genmodel(init_dict)
    meta_params = initialize_meta_params(infer_lr = 0.1, 
                                    nsteps_infer = 1, 
                                    action_lr = 0.1, 
                                    nsteps_action = 1, 
                                    learning_lr = 0.001,
                                    nsteps_learning = 1,
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
    n_timesteps_perturb = len(jnp.arange(start=0, stop=perturb_length, step=dt))

    # get default initializations
    init_dict = get_default_inits(N, perturb_length, dt)

    # initialize generative model, generative process, and meta parameters related to learning and inference
    _, _, genproc, _ = init_gen_process(new_key, init_dict)
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
        return (pos, vel, mu), (pos, vel, F)
    _, history_perturb_nolearn = lax.scan(step_fn, init_state, jnp.arange(n_timesteps_perturb))
    perturb_pos_i, perturb_vel_i, F_responses_nolearn = history_perturb_nolearn[0], history_perturb_nolearn[1], history_perturb_nolearn[2]  # history of perturbed positions, velocities and free energies
    reference_velocity = init_state[1]
    angle_responses_nolearn = compute_turning_magnitudes(reference_velocity, perturb_vel_i)

    fig, axes = plt.subplots(1,2, figsize = (10,8))
    
    for i in range(N-1):
        axes[0].plot(pos_hist[-n_pre:,i,0], pos_hist[-n_pre:,i,1], c = 'b')
        axes[0].plot(perturb_pos_i[:,i,0], perturb_pos_i[:,i,1], c='r')
    axes[0].plot(pos_hist[-n_pre:,-1,0], pos_hist[-n_pre:,-1,1], c = 'b',label='pre-perturbation')
    axes[0].plot(perturb_pos_i[:,-1,0], perturb_pos_i[:,-1,1], c='r',label='post-perturbation')
    axes[0].scatter([perturb_pos_i[0,agent_id,0]], [perturb_pos_i[0,agent_id,1]], s = 50, c = 'k', label = 'Perturbed individual')

    axes[0].set_title('No learning of $\lambda_z$')
    axes[0].legend()

    preparams = {'s_z': jnp.ones(N)}
    parameterization_mapping = {'s_z': {'to_arg_name': 'Pi_z', 'fn': parameterize_Pi_2do}}
    initial_learnable_params = vmap(reparameterize, (0, None))(preparams, parameterization_mapping)
    genmodel = merge(genmodel, initial_learnable_params)
    dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)

    single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)
    # create a custom step function that will run the simulation as you want
    def step_fn(carry, t):
        pos_past, vel_past, mu_past, preparams_past = carry
        pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)
        return (pos, vel, mu, preparams), (pos, vel, F, preparams['s_z'])
    _, history_perturb_learn = lax.scan(step_fn, init_state + (preparams,),  jnp.arange(n_timesteps_perturb))
    perturb_pos_i, perturb_vel_i, F_responses_learn, sz_responses = history_perturb_learn[0], history_perturb_learn[1], history_perturb_learn[2], history_perturb_learn[3] # history of perturbed velocities and free energies
    angle_responses_learn = compute_turning_magnitudes(reference_velocity, perturb_vel_i)


    for i in range(N-1):
        axes[1].plot(pos_hist[-n_pre:,i,0], pos_hist[-n_pre:,i,1], c = 'b')
        axes[1].plot(perturb_pos_i[:,i,0], perturb_pos_i[:,i,1], c='r')
    axes[1].plot(pos_hist[-n_pre:,-1,0], pos_hist[-n_pre:,-1,1], c = 'b',label='pre-perturbation')
    axes[1].plot(perturb_pos_i[:,-1,0], perturb_pos_i[:,-1,1], c='r',label='post-perturbation')
    axes[1].scatter([perturb_pos_i[0,agent_id,0]], [perturb_pos_i[0,agent_id,1]], s = 50, c = 'k',label = 'Perturbed individual')

    axes[1].set_title('Learning of $\lambda_z$ enabled')
    axes[1].legend()
    plt.show()



if __name__ == '__main__':

    # initialize ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', '-s', type = int,
                help = "key to initialize Jax random seed",
                dest = "init_key_num", default=1)
    parser.add_argument('--N', '-N', type = int,
                help = "Number of individuals",
                dest = "N", default=50)
    parser.add_argument('--T', '-T', type = int,
                help = "Duration of simulation (in seconds)",
                dest = "T", default=75)
    parser.add_argument('--perturb_length', '-pl', type = int,
                help = "Length of time in seconds that the simulation is ran following the perturbation",
                dest = "perturb_length", default=10)
    parser.add_argument('--perturb_value', '-pv', type = float,
                help = "Scalar value of perturbation",
                dest = "perturb_value", default=-5.)
    parser.add_argument('--perturb_duration', '-pd', type = int,
                help = "Duration of the perturbation increment itself in timesteps",
                dest = "perturb_duration", default=10)
    parser.add_argument('--n_pre', '-np', type = int,
                help = "Number of timesteps to show trajectories pre-perturbation",
                dest = "n_pre", default=1000)

    args = parser.parse_args()

    run(**vars(args))
