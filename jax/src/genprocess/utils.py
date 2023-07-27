from jax import random, vmap
from jax import numpy as jnp
import math

from . import geometry as geo
from . import default_sensory
from . import get_observations

def init_gen_process(key, init_dict):

    # pull out relevant variables from initialization dict
    N, ns_x, ns_phi, ndo_x, ndo_phi = init_dict['N'], init_dict['ns_x'], init_dict['ns_phi'], init_dict['ndo_x'], init_dict['ndo_phi']
    sector_angles = init_dict['sector_angles']

    pos_vel_init_key, noise_key_obs, noise_key_actions = random.split(key, 3)

    pos, vel = geo.initialize_positions_velocities(pos_vel_init_key, init_dict['N'], **init_dict['posvel_init'])

    assert ns_x == len(sector_angles) - 1, "Number of hidden state dimensions must equal number of visual sectors"
    assert ns_phi == ns_x, "Different dimensionality of observations and hidden states not currently supported"

    genproc = {}
    genproc['R_starts'], genproc['R_ends'] = geo.compute_rotation_matrices(sector_angles, reverse_flag=True)
    genproc['dist_thr'] = init_dict['dist_thr']

    z_gp = jnp.array([math.sqrt(init_dict['z_h']), math.sqrt(init_dict['z_hprime'])]).reshape(1, ndo_phi, 1, 1) # take the square root to turn into a standard deviation (which can be used to simply scale the random samples from N(z;0,1)

    genproc['dt'] = init_dict['dt']
    genproc['T'] = init_dict['T']
    genproc['t_axis'] = jnp.arange(start=0, stop=genproc['T'], step=genproc['dt'])

    noise_tensor = math.sqrt(genproc['dt']) * z_gp * random.normal(noise_key_obs, shape = (len(genproc['t_axis']), ndo_phi, ns_phi, N))

    genproc['sensory_noise'] = noise_tensor
    genproc['sensory_transform'] = default_sensory.identity_transform
    genproc['grad_sensory_transform'] = default_sensory.grad_identity

    genproc['action_noise'] = init_dict['z_action'] * random.normal(noise_key_actions, shape = (len(genproc['t_axis']), N, 2))

    _, new_key = random.split(noise_key_actions)
    return pos, vel, genproc, new_key

def change_noise_variance(noise_tensor, noise_change_t = None, noise_change_scalar = 1.0, do_idx = None, s_idx = None, n_idx = None):
    """ Scale the amplitude of random fluctuations starting at some time `noise_change_t` to the end of the simulation """

    if noise_change_t is None:
        noise_change_t = int(noise_tensor.shape[0]/2)
    
    if do_idx is None:
        do_idx = slice(noise_tensor.shape[1])
    
    if s_idx is None:
        s_idx = slice(noise_tensor.shape[2])

    if n_idx is None:
        n_idx = slice(noise_tensor.shape[3])

    return noise_tensor.at[noise_change_t:,do_idx,s_idx,n_idx].set(noise_change_scalar * noise_tensor[noise_change_t:,do_idx,s_idx,n_idx]) # reduce variance of noise by 10 halfway through




