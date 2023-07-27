from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from funcy import merge
import argparse

from utils import initialize_meta_params, get_default_inits, run_single_simulation
from learning import make_dfdparams_fn, reparameterize
from genprocess import init_gen_process
from genmodel import init_genmodel, create_temporal_precisions
from matplotlib import pyplot as plt

def run(
        init_key_num = 1, # number to initialize the jax random seed
        N = 30, # the number of agents to change per initialization
        T = 100, # how long the simulation should run for (in seconds)
        dt = 0.01, # the time step size for stochastic integration (in seconds)
        average_pi_w=1.0, # the average sensory smoothness across all agents
        save = False # whether to save the results as an npz file to disk
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

    ## Parameterization specific to learning sensory smoothness aka beliefs about higher-order variance of sensory fluctuations
    _, piw_key = random.split(new_key)
    logpiw_spatial_all = jnp.log(average_pi_w) + 0.5 * random.normal(piw_key, shape = (N,)) # sample a different sensory (log) spatial precision for every agent

    Pi_w_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_x'], smoothness=init_dict['s_w']) # technically correct, but need to decrease smoothness s_w to make it look ok

    def parameterize_Pi_w(logpi_w_spatial):
        """
        Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
        """
        Pi_w_spatial = jnp.exp(logpi_w_spatial) * jnp.eye(genmodel['ns_x'])
        return jnp.kron(Pi_w_temporal, Pi_w_spatial)

    # parameterize all the agent-specific generative models with their smoothness parameterizations
    preparams = {'logpiw_spatial': logpiw_spatial_all}
    parameterization_mapping = {'logpiw_spatial': {'to_arg_name': 'Pi_w', 'fn': parameterize_Pi_w}
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

    if save:
        np.savez(f'sim_hist_key{init_key_num}.npz', r=jnp.permute(simulation_history[0], (2, 1, 0)), v=jnp.permute(simulation_history[1], (2, 1, 0)))
    else:
        position_history, sz_history = simulation_history[0], simulation_history[1]['logpiw_spatial']
        
        # plot trajectories
        fig, axes = plt.subplots(1, 2, figsize=(10,8))
        for i in range(N):
            axes[0].plot(position_history[:,i,0], position_history[:,i,1])
        axes[0].set_xlabel('$X$')
        axes[0].set_ylabel('$Y$')

        # plot history of beliefs about sensory smoothness
        axes[1].plot(jnp.arange(0, n_timesteps), sz_history) # plot beliefs about first sector eta for all individuals over time
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Learned beliefs about $\log \pi_w$ over time')

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
                dest = "dt", default=0.01)
    parser.add_argument('--average_pi_w', '-average_pi_w', type = float,
                help = "Average process precision across all agents [default: 1.0]",
                dest = "average_pi_w", default=1.0)
    # add an argument that is a boolean flag for whether to save the results as an npz file to disk (the history of positions and velocities)
    parser.add_argument('--save', '-save', action = 'store_true',
                help = "Whether to save the results as an npz file to disk",
                dest = "save", default=False)

    args = parser.parse_args()

    run(**vars(args))
