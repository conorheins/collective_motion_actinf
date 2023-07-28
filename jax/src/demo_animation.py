from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
import argparse
from funcy import merge

import scipy.signal

from utils import initialize_meta_params, get_default_inits, run_single_simulation, animate_trajectory, str2bool
from genprocess import init_gen_process
from genmodel import init_genmodel

def smooth_trajectories(trajectories, window_size = 5):
    """
    Smooths trajectories using a convolutional window of size `window_size`
    """

    smoothed_traj = np.zeros_like(trajectories)
    conv_window = (np.ones(window_size) / window_size).reshape(-1,1)
    for n in range(trajectories.shape[1]):
        smoothed_traj[:,n,:] = scipy.signal.convolve2d(trajectories[:,n,:], conv_window, mode='full')[:-(window_size-1)] # chop off the end
    return smoothed_traj

def run(init_key_num = 1, # initial key number for the JAX PRNG seed
        N = 30, # number of individuals
        T = 50, # duration of simulation in seconds
        dt = 0.01, # time step size for stochastic integration
        n_sectors = 4, # number of sensory sectors to divide each agent's visual field into
        sector_angle = 60., # angle of each sensory sector in degrees
        init_dict_override = None, # dictionary of parameters to override the default initialization
        meta_params_override = None, # dictionary of parameters to override the default meta parameters
        start_t = 0, # starting timestep for the animation
        end_t = 100, # ending timestpe for the animation
        min_size=0.0, # minimum size of the individuals
        max_size=8.0, # maximum size of the individuals
        skip = 1, # how many timesteps to skip in between frames
        t_steps_back = 5, # how long each trajectory should be shown back in time before vanishing
        smooth_flag = False, # boolean for whether to smooth the trajectory or not
        window_size = 5, # size of convolutional window for smoothing
        frames_per_second = 20, # frames per second for the animation
        save_name = None # name of the file to save the .gif to
        ):

        key = random.PRNGKey(init_key_num)
        init_dict = get_default_inits(N, T, dt, n_sectors=n_sectors, sector_angle=sector_angle)
        if init_dict_override is not None:
            init_dict = merge(init_dict, init_dict_override)

       # initialize generative model, generative process, and meta parameters related to learning and inference
        pos, vel, genproc, new_key = init_gen_process(key, init_dict)
        genmodel = init_genmodel(init_dict)

        meta_params = initialize_meta_params(**meta_params_override)

        n_timesteps = len(genproc['t_axis'])

        if start_t < 0:
            start_t = n_timesteps + start_t
        if end_t < 0:
            end_t = n_timesteps + end_t

        # initialize first beliefs using priors over hidden states
        mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T
        init_state = (pos, vel, mu)

        history = run_single_simulation(init_state, n_timesteps, genmodel, genproc, meta_params, returns = ['pos'], learning=False)

        if smooth_flag:
            r = smooth_trajectories(history, window_size=window_size)
        else:
            r = np.array(history)

        final_save_name = animate_trajectory(r, start_t=start_t, end_t=end_t, skip=skip, min_size=min_size, max_size=max_size, t_steps_back=t_steps_back, fps=frames_per_second, save_name=save_name)
        print(f'Animation saved to {final_save_name}\n')

if __name__ == '__main__':

    # initialize ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', '-s', type = int,
                help = "key to initialize Jax random seed",
                dest = "init_key_num", default=1)
    parser.add_argument('--N', '-N', type = int,
                help = "Number of individuals",
                dest = "N", default=30)
    parser.add_argument('--T', '-T', type = int,
                help = "Duration of simulation (in seconds)",
                dest = "T", default=50)
    parser.add_argument('--dt', '-dt', type = float,
                help = "Time step size for stochastic integration",
                dest = "dt", default=0.01)
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
    parser.add_argument('--start_t', '-st', type = int,
                help = "Start timestep (in absolute time) of animation",
                dest = "start_t", default=0)
    parser.add_argument('--end_t', '-et', type = int,
                help = "End timestep (in absolute simulation time) animation",
                dest = "end_t", default=100)
    parser.add_argument('--min_size', '-mins', type = float,
                help = "Minimum size of particles in animation",
                dest = "min_size", default=0.)
    parser.add_argument('--max_size', '-maxs', type = float,
                help = "Maximum size of particles in animation",
                dest = "max_size", default=8.)
    parser.add_argument('--skip', '-sk', type = int,
                help = "Skip timestep of animation",
                dest = "skip", default=1)
    parser.add_argument('--t_steps_back', '-tsb', type = int,
                help = "Number of timesteps back to show the trajectory of each particle",
                dest = "t_steps_back", default=5)
    parser.add_argument( '--smooth_flag', '-sm',
                        action='store_true', help='Whether to smooth trajectories')
    parser.add_argument('--window_size', '-ws', type = int,
                help = "Size of smoothing window in timesteps",
                dest = "window_size", default=5)
    parser.add_argument('--frames_per_second', '-fps', type = int,
                help = "Frames per second to render the gif",
                dest = "frames_per_second", default=20)
    parser.add_argument('--save_name', '-svnam', type = str,
                help = "Name of the file to save the gif to",
                dest = "save_name", default=None)
    args = parser.parse_args()

    # get the arguments to init_dict_override as a dictionary
    init_dict_override = {k:v for k,v in vars(args).items() if k in ['dist_thr', 'z_h', 'z_hprime', 'z_action', 'alpha', 'eta', 'pi_z_spatial', 'pi_w_spatial', 's_z', 's_w']}

    # get the arguments to meta_params_override as a dictionary
    meta_params_override = {k:v for k,v in vars(args).items() if k in ['infer_lr', 'nsteps_infer', 'action_lr', 'nsteps_action', 'normalize_v']}

    # get all the remaining arguments as a dictionary
    common_args = {k:v for k,v in vars(args).items() if k in ['init_key_num', 'N', 'T', 'dt', 'n_sectors', 'sector_angle', 'start_t', 'end_t', 'min_size', 'max_size', 'skip', 't_steps_back', 'smooth_flag', 'window_size', 'frames_per_second', 'save_name']}
    
    run(**common_args, init_dict_override=init_dict_override, meta_params_override=meta_params_override)