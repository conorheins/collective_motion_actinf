from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import argparse
from funcy import merge

from utils import initialize_meta_params, get_default_inits, run_single_simulation, str2bool
from genprocess import init_gen_process
from genmodel import init_genmodel


def run(
        init_key_num = 1, # number to initialize the jax random seed
        N = 30, # the number of agents to change per initialization
        T = 100, # how long the simulation should run for (in seconds)
        dt = 0.01, # the time step size for stochastic integration (in seconds)
        n_sectors = 4, # number of sensory sectors to divide each agent's visual field into
        sector_angle = 60., # angle of each sensory sector in degrees
        last_T_seconds = 10, # how long from the last timestep backwards, to plot 
        save = False, # whether to save the results as an npz file to disk
        init_dict_override = None, # dictionary of parameters to override the default initialization
        meta_params_override = None # dictionary of parameters to override the default meta parameters
        ):

    seed_key = random.PRNGKey(init_key_num)

    # set up some global parameters
    key = random.PRNGKey(init_key_num)
    init_dict = get_default_inits(N, T, dt, n_sectors=n_sectors, sector_angle=sector_angle)
    if init_dict_override is not None:
        init_dict = merge(init_dict, init_dict_override)
    
    # initialize generative model, generative process, and meta parameters related to learning and inference
    pos, vel, genproc, new_key = init_gen_process(key, init_dict)
    genmodel = init_genmodel(init_dict)

    meta_params = initialize_meta_params(**meta_params_override)

    # initialize first beliefs using priors over hidden states
    init_mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

    n_timesteps = len(genproc['t_axis'])
    init_state = (pos, vel, init_mu)

    vars2return = ['pos','vel'] if save else ['pos']

    simulation_history = run_single_simulation(init_state, n_timesteps, genmodel, genproc, meta_params, returns = vars2return, learning=False)

    if save:
        np.savez(f'sim_hist_key{init_key_num}.npz', r=simulation_history[0], v=simulation_history[1])
    else:
        last_t_dt = int(last_T_seconds/dt)
        position_history = simulation_history
        fig, ax = plt.subplots(figsize=(10,8))
        for i in range(N):
            ax.plot(position_history[-last_t_dt:,i,0], position_history[-last_t_dt:,i,1])
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
                dest = "dt", default=0.01)
    parser.add_argument('--last_T_seconds', '-lastT', type = float,
                help = "How many seconds to plot from the end of the simulation",
                dest = "last_T_seconds", default=10)
    parser.add_argument('--n_sectors', '-nsec', type = int,
                help = "Number of sensory sectors to divide each agent's visual field into",
                dest = "n_sectors", default=4)
    parser.add_argument('--sector_angle', '-secang', type = float,
                help = "Angle of each sensory sector in degrees",
                dest = "sector_angle", default=60.0)
    parser.add_argument('--dist_thr', '-dthr', type = float,
                help = "Cut-off within which neighbouring agents are detected",
                dest = "dist_thr", default=5.0)
    parser.add_argument('--z_h', '-zh', type = float,
                help = "Variance of additive observation noise on first order hidden states",
                dest = "z_h", default=0.01)
    parser.add_argument('--z_hprime', '-zhp', type = float, 
                help = "Variance of additive observation noise on second order hidden states (velocity observations)",
                dest = "z_hprime", default=0.01)
    parser.add_argument('--z_action', '-za', type = float,
                help = "Variance of movement/action (additive noise onto x/y components of velocity vector during stochastic integration)",
                dest = "z_action", default=0.01)
    parser.add_argument('--alpha', '-alpha', type = float,
                help = "Strength of flow function (the decay coefficient in case of independent dimensions)",
                dest = "alpha", default=0.5)
    parser.add_argument('--eta', '-eta', type = float,
                help = "The fixed point of the flow function",
                dest = "eta", default=1.)
    parser.add_argument('--pi_z_spatial', '-piz', type = float,
                help = "The spatial variance of the sensory precision",
                dest = "pi_z_spatial", default=1.0)
    parser.add_argument('--pi_w_spatial', '-piw', type = float,
                help = "The spatial variance of the model or process precision",
                dest = "pi_w_spatial", default=1.0)
    parser.add_argument('--s_z', '-s_z', type = float,
                help = "The assumed smoothness (temporal autocorrelation) of sensory fluctuations",
                dest = "s_z", default=1.0)
    parser.add_argument('--s_w', '-s_w', type = float,
                help = "The assumed smoothness (temporal autocorrelation) of process fluctuations",
                dest = "s_w", default=1.0)
    parser.add_argument('--infer_lr', '-infer_lr', type = float,
                help = "Learning rate for inference",
                dest = "infer_lr", default=0.1)
    parser.add_argument('--nsteps_infer', '-nsteps_infer', type = int,
                help = "Number of inference steps per time step",
                dest = "nsteps_infer", default=1)
    parser.add_argument('--action_lr', '-action_lr', type = float,
                help = "Learning rate for action",
                dest = "action_lr", default=0.1)
    parser.add_argument('--nsteps_action', '-nsteps_action', type = int,
                help = "Number of action steps per time step",
                dest = "nsteps_action", default=1)
    parser.add_argument('--normalize_v', '-nv', type=str2bool,
                nargs='?', const=True, help = "Whether to normalize the velocity vectors of all agents at each time step to unit magnitude",
                dest = "normalize_v", default=True)
    # add an argument that is a boolean flag for whether to save the results as an npz file to disk (the history of positions and velocities)
    parser.add_argument('--save', '-save', action = 'store_true',
                help = "Whether to save the results as an npz file to disk",
                dest = "save", default=False)

    args = parser.parse_args()

    # get the arguments to init_dict_override as a dictionary
    init_dict_override = {k:v for k,v in vars(args).items() if k in ['dist_thr', 'z_h', 'z_hprime', 'z_action', 'alpha', 'eta', 'pi_z_spatial', 'pi_w_spatial', 's_z', 's_w']}

    # get the arguments to meta_params_override as a dictionary
    meta_params_override = {k:v for k,v in vars(args).items() if k in ['infer_lr', 'nsteps_infer', 'action_lr', 'nsteps_action', 'normalize_v']}

    # get all the remaining arguments as a dictionary
    common_args = {k:v for k,v in vars(args).items() if k in ['init_key_num', 'N', 'T', 'dt', 'n_sectors', 'sector_angle', 'last_T_seconds', 'save']}
    
    run(**common_args, init_dict_override=init_dict_override, meta_params_override=meta_params_override)


