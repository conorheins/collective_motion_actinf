from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from funcy import merge
import argparse

from utils import initialize_meta_params, get_default_inits, run_single_simulation, animate_trajectory, is_connected_over_time
from learning import make_dfdparams_fn, reparameterize
from genprocess import init_gen_process
from genmodel import init_genmodel, create_temporal_precisions
from matplotlib import pyplot as plt

def run(
        init_key_num = 1, # number to initialize the jax random seed
        N = 30, # the number of agents to change per initialization
        T = 100, # how long the simulation should run for (in seconds)
        dt = 0.01, # the time step size for stochastic integration (in seconds)
        last_T_seconds = 10, # how long from the last timestep backwards, to plot 
        average_logpiz=0.0, # the average sensory smoothness across all agents
        average_logpiw=0.0, # the average process smoothness across all agents
        save = False, # whether to save the results as an npz file to disk
        remove_strays = False, # where to save the results
        animate = False, # whether to animate the results
        distance_threshold = 5.0   # threshold to use when deciding to remove strays from visualization
        ):

    # set up some global stuff (random key, T, dt, N, D)
    key = random.PRNGKey(init_key_num)
    init_dict = get_default_inits(N, T, dt)

    # initialize generative model, generative process, and meta parameters related to learning and inference
    pos, vel, genproc, new_key = init_gen_process(key, init_dict)
    genmodel = init_genmodel(init_dict)
    meta_params = initialize_meta_params(infer_lr = 0.1, 
                                        nsteps_infer = 1, 
                                        action_lr = 0.1, 
                                        nsteps_action = 1, 
                                        learning_lr = 0.001, 
                                        nsteps_learning = 1, 
                                        normalize_v = True
                                        )

    piz_key, piw_key = random.split(new_key)

    Pi_z_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_phi'], smoothness=init_dict['s_z']) # technically correct, but need to decrease smoothness s_z to make it look ok
    logpi_z_spatial_all = random.uniform(piz_key, minval=average_logpiz-0.05, maxval=average_logpiz+0.05, shape = (N,genmodel['ns_phi'])) # sample a different sensory precision for every sector, for every agent

    def parameterize_Pi_z(logpiz_spatial):
        """
        Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
        """
        
        Pi_z_spatial = jnp.diag(jnp.exp(logpiz_spatial)) # multivariate pi_z_spatial
        return jnp.kron(Pi_z_temporal, Pi_z_spatial)
    
    Pi_w_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_x'], smoothness=init_dict['s_w']) # technically correct, but need to decrease smoothness s_z to make it look ok
    logpi_w_spatial_all = random.uniform(piw_key, minval = average_logpiw-0.05, maxval=average_logpiw+0.05, shape = (N,genmodel['ns_x'])) # sample a different sensory precision for every sector, for every agent

    def parameterize_Pi_w(logpiw_spatial):
        """
        Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
        """
        
        Pi_w_spatial = jnp.diag(jnp.exp(logpiw_spatial)) # multivariate pi_z_spatial
        return jnp.kron(Pi_w_temporal, Pi_w_spatial)

    preparams = {'logpi_z_spatial': logpi_z_spatial_all, 'logpi_w_spatial': logpi_w_spatial_all}

    parameterization_mapping = {'logpi_z_spatial': {'to_arg_name': 'Pi_z', 'fn': parameterize_Pi_z},
                                'logpi_w_spatial': {'to_arg_name': 'Pi_w', 'fn': parameterize_Pi_w}
                                    }                                
    
    initial_learnable_params = vmap(reparameterize, (0, None))(preparams, parameterization_mapping)
    genmodel = merge(genmodel, initial_learnable_params)

    dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
    learning_args = {'dFdparam': dFdparam_function, 'param_mapping': parameterization_mapping}

    # initialize first beliefs using priors over hidden states
    init_mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T
    n_timesteps = len(genproc['t_axis'])
    init_state = (pos, vel, init_mu, preparams)

    vars2return = ['pos','vel','preparams'] if save else ['pos', 'preparams']

    # Run the simulation with learning enabled
    simulation_history  = run_single_simulation(init_state, n_timesteps, genmodel, genproc, meta_params, returns = vars2return, learning=True, learning_args=learning_args)

    position_history = simulation_history[0]
    if save:
        np.savez(f'sim_hist_key{init_key_num}.npz', r=position_history, v=simulation_history[1])
    else:
        position_history, logpiz_history, logpiw_history = simulation_history[0], simulation_history[1]['logpi_z_spatial'], simulation_history[1]['logpi_w_spatial']
        
        if remove_strays:
            centroid = jnp.mean(position_history[-1], axis=0)
            dists = jnp.linalg.norm(position_history[-1] - centroid, axis=1)
            good_indices = jnp.where(dists < distance_threshold)[0]

            position_history = position_history[:,good_indices,:]
            logpiz_history = logpiz_history[:,good_indices,:]
            logpiw_history = logpiw_history[:,good_indices,:]
            N_reduced = len(good_indices)
        else:
            N_reduced = N


        last_t_dt = int(last_T_seconds/dt)
        is_connected_hist = np.array(is_connected_over_time(position_history[-last_t_dt:], thr=distance_threshold))

        # plot trajectories
        fig, axes = plt.subplots(1, 4, figsize=(10,8))
        for i in range(N_reduced):
            last_t_pos = position_history[-last_t_dt:,i,:]
            axes[0].plot(last_t_pos[is_connected_hist,0], last_t_pos[is_connected_hist,1], color='b')
            axes[0].plot(last_t_pos[~is_connected_hist,0], last_t_pos[~is_connected_hist,1], color='r')
            # axes[0].plot(position_history[-last_t_dt:,i,0], position_history[-last_t_dt:,i,1])
        axes[0].set_xlabel('$X$')
        axes[0].set_ylabel('$Y$')

        # plot history of beliefs about sensory smoothness
        axes[1].plot(jnp.arange(n_timesteps-last_t_dt, n_timesteps), logpiz_history[-last_t_dt:,0,:]) 
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Learned beliefs about $\log \pi_z$ over time')

        # plot history of beliefs about process smoothness
        axes[2].plot(jnp.arange(n_timesteps-last_t_dt, n_timesteps), logpiw_history[-last_t_dt:,0,:]) 
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Learned beliefs about $\log \pi_{\omega}$ over time')

        
        axes[3].plot(jnp.arange(n_timesteps-last_t_dt, n_timesteps)[is_connected_hist], is_connected_hist[is_connected_hist], color='b', label = 'connected')
        axes[3].plot(jnp.arange(n_timesteps-last_t_dt, n_timesteps)[~is_connected_hist], is_connected_hist[~is_connected_hist], color='r', label = 'disconnected')
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Group connectedness over time')
        axes[3].legend()
 
        plt.show()

    if remove_strays:
        centroid = jnp.mean(position_history[-1], axis=0)
        dists = jnp.linalg.norm(position_history[-1] - centroid, axis=1)
        good_indices = jnp.where(dists < distance_threshold)[0]
        position_history = position_history[:,good_indices,:]

    if animate:
        final_save_name = animate_trajectory(np.array(position_history), start_t=len(genproc['t_axis']) - 2000, end_t=len(genproc['t_axis'])-1, skip=25, t_steps_back=12, fps=30)
        print(f'Animation saved to {final_save_name}\n')

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
                dest = "dt", default=0.01)
    parser.add_argument('--last_T_seconds', '-lastT', type = float,
                help = "How many seconds to plot from the end of the simulation",
                dest = "last_T_seconds", default=10)
    parser.add_argument('--average_logpiz', '-average_logpiz', type = float,
                help = "Average log sensory precision",
                dest = "average_logpiz", default=0.0)
    parser.add_argument('--average_logpiw', '-average_logpiw', type = float,
                help = "Average log process precision",
                dest = "average_logpiw", default=0.0)
    # add an argument that is a boolean flag for whether to save the results as an npz file to disk (the history of positions and velocities)
    parser.add_argument('--save', '-save', action = 'store_true',
                help = "Whether to save the results as an npz file to disk",
                dest = "save", default=False)
    parser.add_argument('--remove_strays', '-remove_strays', action = 'store_true',
                help = "Whether to remove agents that stray too far from the center of the swarm",
                dest = "remove_strays", default=False)
    parser.add_argument('--animate', '-ani', action = 'store_true',
                help = "Whether to save an animation of the simulation",
                dest = "animate", default=False)
    parser.add_argument('--distance_threshold', '-dthr', type = float,
                help = "Distance threshold for removing agents that stray too far from the center of the swarm",
                dest = "distance_threshold", default=5.0)

    args = parser.parse_args()

    run(**vars(args))
