from jax import random, vmap
from jax import numpy as jnp
import numpy as np

from couzin_2zone import simulate_trajectories
import matplotlib.pyplot as plt
import argparse

def run(init_key_num = 1, # number to initialize the jax random seed
        N = 30, # the number of agents to change per initialization
        T = 100, # how long the simulation should run for (in seconds)
        dt = 0.1, # the time step size for stochastic integration (in seconds)
        fop_angle = 315.0, # the field of perception angle (in degrees)
        dist_thr = 10.0, # the distance threshold for the field of perception (in cm)
        repulsion_thr = 1.0, # the distance threshold for repulsion (in cm)
        speed = 1.0, # the speed of the agents (in units/s)
        angular_turning_rate = 40.0, # the angular turning rate of the agents (in degrees/s)
        angular_threshold = 20.0, # the angular threshold for the field of perception (in degrees)
        z_action = 0.001, # the standard deviation of the action noise (in units/s^2)
        debug_mode = False, # whether to return all the intermediate data arrays for debugging purposes
        save = False, # whether to save the results as an npz file to disk
        n_inits = 1 # the number of parallel initialization to run
        ):
    key = random.PRNGKey(init_key_num)
    
    # calculate number of timesteps, given T and dt
    n_timesteps = int(T/dt)

    particle_params = {'fop_angle': fop_angle, 'dist_thr': dist_thr, 'repulsion_thr': repulsion_thr, 'speed': speed, 'angular_turning_rate': angular_turning_rate, 'angular_threshold': angular_threshold}

    if n_inits == 1:

        pos = random.uniform(key, minval = -1.0, maxval = 1.0, shape = (N,2))
        _, key = random.split(key)
        vel = random.uniform(key, minval = 0.5, maxval = 1.0, shape = (N,2))
        vel = vel / jnp.linalg.norm(vel, axis = 1, keepdims = True)

        noise_params = {'action_noise': z_action*random.normal(key, shape=(n_timesteps, N, 2))}

        init_state = (pos, vel)
        all_outs = simulate_trajectories(init_state, noise_params, particle_params, dt=dt, n_timesteps=n_timesteps, debug=debug_mode)
    else:
        
        def run_single_init(init_i_key):
            pos = random.uniform(init_i_key, minval = -1.0, maxval = 1.0, shape = (N,2))
            _, init_i_key = random.split(init_i_key)
            vel = random.uniform(init_i_key, minval = 0.5, maxval = 1.0, shape = (N,2))
            vel = vel / jnp.linalg.norm(vel, axis = 1, keepdims = True)

            noise_params = {'action_noise': z_action*random.normal(init_i_key, shape=(n_timesteps, N, 2))}

            init_state = (pos, vel)
            all_outs = simulate_trajectories(init_state, noise_params, particle_params, dt=dt, n_timesteps=n_timesteps, debug=debug_mode)

            return all_outs
        
        all_outs = vmap(run_single_init)(random.split(key, n_inits))
    
    if save:
        np.savez(f'couzin2zone_sim_hist_key{init_key_num}.npz', r=all_outs[0], v=all_outs[1])
    else:
        if n_inits == 1:
            pos_hist, vel_hist = all_outs[0], all_outs[1]
        else:
            pos_hist, vel_hist = all_outs[0][0], all_outs[1][0]
        
        fig, ax = plt.subplots(figsize=(10,8))
        for i in range(N):
            ax.plot(pos_hist[:,i,0], pos_hist[:,i,1])
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')

        plt.show()

if __name__ == '__main__':

    # initialize ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', '-s', type = int,
                help = "key to initialize Jax random seed",
                dest = "init_key_num", default=1)
    parser.add_argument('--N', '-N', type = int,
                help = "Number of agents",
                dest = "N", default=30)   
    parser.add_argument('--T', '-T', type = float,
                help = "Number of seconds to run simulation",
                dest = "T", default=100)    
    parser.add_argument('--dt', '-dt', type = float,
                help = "Time step size for stochastic integration",
                dest = "dt", default=0.1)
    parser.add_argument('--fop_angle', '-fop_angle', type = float,
                help = "Field of perception angle (in degrees)",
                dest = "fop_angle", default=315.0)
    parser.add_argument('--dist_thr', '-dist_thr', type = float,
                help = "Distance threshold for field of perception (in cm)",
                dest = "dist_thr", default=10.0)
    parser.add_argument('--repulsion_thr', '-repulsion_thr', type = float, 
                help = "Distance threshold for repulsion (in cm)",
                dest = "repulsion_thr", default=1.0)
    parser.add_argument('--speed', '-speed', type = float,
                help = "Speed of agents (in units/s)",
                dest = "speed", default=1.0)
    parser.add_argument('--angular_turning_rate', '-atr', type = float, 
                help = "Angular turning rate of agents (in degrees/s)",
                dest = "angular_turning_rate", default=40.0)
    parser.add_argument('--angular_threshold', '-ath', type = float,
                help = "Angular threshold for field of perception (in degrees)",
                dest = "angular_threshold", default=20.0)
    parser.add_argument('--z_action', '-z_action', type = float,
                help = "Standard deviation of action noise (in units/s^2)",
                dest = "z_action", default=0.001)
    parser.add_argument('--debug', '-debug', type = bool,
                help = "Debug mode",
                dest = "debug_mode", default=False) 
    # add an argument that is a boolean flag for whether to save the results as an npz file to disk (the history of positions and velocities)
    parser.add_argument('--save', '-save', action = 'store_true',
                help = "Whether to save the results as an npz file to disk",
                dest = "save", default=False)
    parser.add_argument('--n_inits', '-ni', type = int,
                help = "Number of parallel realizations to run",
                dest = "n_inits", default=2) 

    args = parser.parse_args()

    run(**vars(args))