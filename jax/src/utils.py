from funcy import merge
from jax import lax, vmap, jit, grad, random
from jax import numpy as jnp
from jax import scipy as jsp

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import os
import time

from genprocess import get_observations, get_observations_special, advance_positions, init_gen_process, compute_Dgroup_and_rankings_t, compute_Dgroup_and_rankings_vmapped, compute_turning_magnitudes, compute_integrated_change_magnitude
from inference import run_inference
from genmodel import compute_vfe_vectorized
from action import infer_actions
from learning import update_parameters, reparameterize, make_dfdparams_fn
from couzin_2zone import simulate_trajectories

def make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params):
    """
    Function that creates and returns the single timestep function used to simulate a collective AIF simulation, 
    given the generative process, generative model, and other parameters
    """

    def single_timestep(pos, vel, mu, preparams, t_idx):

        # sample observations from generative process
        phi, all_dh_dr_self, empty_sectors_mask = get_observations(pos, vel, genproc, t_idx)

        # update the generative model 
        learned_params = vmap(reparameterize, (0, None))(preparams, parameterization_mapping)
        genmodel_t = merge(genmodel, learned_params) # update generative model using new preparams

        # run hidden state inference 
        infer_res, mu_traj = run_inference(phi, mu, empty_sectors_mask, genmodel_t, **meta_params['inference_params'])
        mu_next, epsilon_z = infer_res

        # compute variational free energy 
        F = compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel_t)

        # use results of inference to update actions
        vel_next = infer_actions(vel, epsilon_z, genmodel_t, all_dh_dr_self, **meta_params['action_params'])

        # use actions to update generative process
        pos_next = advance_positions(pos, vel_next, genproc['action_noise'][t_idx], dt = genproc['dt'])

        # update generative model parameters
        preparams_next = update_parameters(phi, mu, empty_sectors_mask, preparams, dFdparam_function, **meta_params['learning_params'])

        return pos_next, vel_next, mu_next, preparams_next, F
    
    return single_timestep

def make_single_timestep_fn_nolearning(genproc, genmodel, meta_params):
    """
    Function that creates and returns the single timestep function used to simulate a collective AIF simulation, 
    given the generative process, generative model, and other parameters.
    In this version parameters of the generative model are not learned over time.
    """

    def single_timestep(pos, vel, mu, t_idx):

        # sample observations from generative process
        phi, all_dh_dr_self, empty_sectors_mask = get_observations(pos, vel, genproc, t_idx)

        # run hidden state inference 
        infer_res, mu_traj = run_inference(phi, mu, empty_sectors_mask, genmodel, **meta_params['inference_params'])
        mu_next, epsilon_z = infer_res

        # compute variational free energy 
        F = compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel)

        # use results of inference to update actions
        vel_next = infer_actions(vel, epsilon_z, genmodel, all_dh_dr_self, **meta_params['action_params'])

        # use actions to update generative process
        pos_next = advance_positions(pos, vel_next, genproc['action_noise'][t_idx], dt = genproc['dt'])

        return pos_next, vel_next, mu_next, F
    
    return single_timestep

def initialize_meta_params(infer_lr = 0.1, nsteps_infer = 1, action_lr = 0.1, nsteps_action = 1, learning_lr = 0.001, nsteps_learning = 1, normalize_v = True):

    meta_params = {
    'inference_params': {'k_mu': infer_lr, 
                    'num_steps': nsteps_infer
                },
    'action_params':{'k_alpha': action_lr,
                    'num_steps': nsteps_action,
                    'normalize_v': normalize_v
                },
    'learning_params': {'k_params': learning_lr,
                    'num_steps': nsteps_learning
                }
    }

    return meta_params

def get_default_inits(N, T, dt, n_sectors=4, sector_angle=60.0):

    half_sectors = n_sectors // 2
    # Get the angles for the sectors on one side of 0.
    angles_positive = [sector_angle * i for i in range(half_sectors, -1, -1)]
    # Get the angles for the sectors on the other side of 0, counting backward from 360.
    angles_negative = [360 - (sector_angle * i) for i in range(1, half_sectors + 1)]
    # Concatenate the two lists.
    sector_angles = angles_positive + angles_negative
    assert len(sector_angles) == (n_sectors+1)

    default_init_dict = {'N': N,
                        'posvel_init': {'pos_x_bounds': [-1., 1.],
                                        'pos_y_bounds': [-1., 1.],
                                        'vel_x_bounds': [-1., 1.],
                                        'vel_x_bounds': [-1., 1.],
                                        },
                        'T': T, # total length of simulation (in seconds)
                        'dt': dt, # duration of integration timestep for stochastic integration (in seconds)
                        'sector_angles': sector_angles, # angles of visual sectors
                        # 'sector_angles': [240., 120., 0., 360. - 120., 360. - 240.], # angles of visual sectors
                        'ns_x': n_sectors, # dimensionality of hidden states
                        'ndo_x': 3, # number of dynamical orders of hidden states
                        'ns_phi': n_sectors, # dimensionality of observations
                        'ndo_phi': 2, # number of dynamical orders of observations
                        'dist_thr': 5.0, # cut-off within which neighbouring agents are detected
                        'z_h': 0.01,      # variance of additive observation noise on first order hidden states
                        'z_hprime': 0.01, # variance of additive observation noise on second order hidden states ("velocity" observations)
                        'z_action': 0.01, # variance of movement/action (additive noise onto x/y components of velocity vector during stochastic integration),
                        'alpha': 0.5,   # strength of flow function (the decay coefficient in case of independent dimensions)
                        'eta': 1.,       # the fixed point of the flow function
                        'pi_z_spatial': 1.0, # the spatial variance of the sensory precision
                        'pi_w_spatial': 1.0, # the spatial variance of the model or "process" precision
                        's_z': 1.0,          # the assumed smoothness (temporal autocorrelation) of sensory fluctuations
                        's_w': 1.0           # the assumed smoothness (temporal autocorrelation) of process fluctuations
    }   

    return default_init_dict

def smooth_ts(timeseries, window_size = 5):
    """ Smooth trajectories of shape (T, D) with a window of size `window_size`. """
    conv_window = (jnp.ones(window_size) / window_size).reshape(-1,1)
    smoothed_ts = jsp.signal.convolve2d(timeseries, conv_window, mode='full')[:-(window_size-1)] # chop off the end
    return smoothed_ts

def parameterize_stim_pattern(n_repeats=12, t_per_pattern=80, amplitude=15.0, direction='right'):
    """ Parameterizes a noise stimulation structure that will be copied across perturbed individuals.
    It repeats `n_repeats` times, and moves across the four sectors going from left --> right or right --> left.
    The amplitude of the stimulation is `amplitude` and the duration of each stimulation pattern repeat is `t_per_patter` """

    pulse_duration = int(t_per_pattern/4)
    amplitude_pattern = jnp.diag(jnp.array([-0.5, -1.0, 1.0, 0.5])) # left to right pattern
    amptliude_pattern = -1.0 * amplitude_pattern if direction == 'left' else amplitude_pattern
    noise_structure = jnp.kron(amplitude_pattern, jnp.ones((pulse_duration,1)))
    noise_structure = jnp.vstack(n_repeats*[noise_structure])
    noise_structure = smooth_ts(noise_structure, window_size = pulse_duration)

    return amplitude * noise_structure

def animate_trajectory(r, 
                       start_t = 0, 
                       end_t = 100, 
                       skip = 1, 
                       min_size=0., 
                       max_size=8., 
                       t_steps_back=10,
                       fps = 20,
                       save_dir = ".", 
                       save_name = None
                       ):

    n_timesteps, N = r.shape[0], r.shape[1]

    end_t = n_timesteps if end_t > n_timesteps else end_t

    t_indices = np.arange(start_t, end_t, step=skip, dtype = int)

    # pre-generate `segment_indices` to save computation time
    segment_indices = {t: np.arange(max(0, t - skip*t_steps_back), t, step=skip, dtype=int) for t in t_indices}
    
    lwidths = np.linspace(min_size,max_size,t_steps_back-1)
    colors = cm.viridis(np.linspace(0.4,0.9,num=(t_steps_back-1)))[::-1]
    fig, ax = plt.subplots(figsize=(16,16))

    # try using an init function like in the vicsek demo
    x_bounds = [np.min(r[t_indices,:,0]), np.max(r[t_indices,:,0])]
    y_bounds = [np.min(r[t_indices,:,1]), np.max(r[t_indices,:,1])]
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)

    for spine in ax.spines.values():
        spine.set_visible(False)

    def animate(t):
        ax.cla() # clear the previous image
        points_all = np.expand_dims(r[segment_indices[t]], 1)
        segments_all = np.concatenate([points_all[:-1], points_all[1:]], axis = 1) # stack the beginning (t) and end (t+1) of each line-segment, across all agents
        segments_all_list = np.split(segments_all,N, axis=2) # one array of size (t_steps_back-1, 1, 2) per particle, in a list of such arrays
        lc = LineCollection(np.vstack(segments_all_list).squeeze(), colors = colors, linewidths=lwidths)
        collection = ax.add_collection(lc)
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for spine in ax.spines.values():
            spine.set_visible(False)
        # ax.autoscale_view() # comment this out and try the init_fn instead

    anim = FuncAnimation(fig, animate, frames = t_indices, interval = 1, blit = False)
    # anim = FuncAnimation(fig, animate, frames = t_indices, init_func=init_fn, interval=1, blit=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_name is None:
        # save_name = f'{N}_particles_{len(t_indices)}_ts_{timestr}.gif'
        save_name = f'{N}_particles_{len(t_indices)}_ts_{timestr}.mp4'

    
    full_save_path = os.path.join(save_dir, save_name)
    # anim.save(full_save_path, writer='imagemagick', fps=fps)
    anim.save(full_save_path, writer='ffmpeg', fps=fps)

    return full_save_path

def animate_trajectories_compare(r_list,
                                 dt = 0.01,
                                 start_t = 0,
                                 end_t = 1500,
                                 perturb_start_t = 0,
                                 skip = 1, 
                                 min_size=0., 
                                 max_size=8., 
                                 t_steps_back=10,
                                 agents_to_highlight = None,
                                 fps = 20,
                                 save_dir = ".", 
                                 save_name = None
                                ):
    
    n_timesteps, N = r_list[0].shape[0], r_list[0].shape[1]
    end_t = n_timesteps if end_t > n_timesteps else end_t

    t_indices = np.arange(start_t, end_t, step=skip, dtype = int)

    # pre-generate `segment_indices` to save computation time
    segment_indices = {t: np.arange(max(0, t - skip*t_steps_back), t, step=skip, dtype=int) for t in t_indices}
    
    lwidths = np.linspace(min_size,max_size,t_steps_back-1)
    colors = cm.viridis(np.linspace(0.4,0.9,num=(t_steps_back-1)))[::-1]

    highlight_colors = cm.Reds(np.linspace(0.4,0.9,num=(t_steps_back-1)))[::-1]
    highlight_lwidths = {t: lwidths.copy() for t in t_indices}
    # for t in t_indices:
    #     if t >= perturb_start_t:
    #         highlight_lwidths[t] = 2 * lwidths

    fig, axes = plt.subplots(1, len(r_list), figsize=(20,12))

    bounds_list = []
    for ax, r in zip(axes, r_list):
        xbounds = [np.min(r[t_indices,:,0]), np.max(r[t_indices,:,0])]
        ybounds = [np.min(r[t_indices,:,1]), np.max(r[t_indices,:,1])]
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)
        bounds_list.append([xbounds, ybounds])

        for spine in ax.spines.values():
            spine.set_visible(False)

    offset = 400 # add an offset because of the processing time it takes for the agent to transmit the information
    if agents_to_highlight is None:

        def animate(t):

            for ax, r, bounds  in zip(axes, r_list, bounds_list):
                ax.cla() # clear the previous image
                points_all = np.expand_dims(r[segment_indices[t],:,:], 1)
                segments_all = np.concatenate([points_all[:-1], points_all[1:]], axis = 1) # stack the beginning (t) and end (t+1) of each line-segment, across all agents
                segments_all_list = np.split(segments_all,N, axis=2) # one array of size (t_steps_back-1, 1, 2) per particle, in a list of such arrays
                lc = LineCollection(np.vstack(segments_all_list).squeeze(), colors = colors, linewidths=lwidths)
                collection = ax.add_collection(lc)
                # ax.set_xlim(bounds[0])
                # ax.set_ylim(bounds[1])
                ax.set_xlim(bounds_list[1][0])
                ax.set_ylim(bounds_list[1][1])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                for spine in ax.spines.values():
                    spine.set_visible(False)

    else:
        other_ids = np.setdiff1d(np.arange(N), agents_to_highlight)
        num_other = len(other_ids)

        if isinstance(agents_to_highlight, int):
            num_special = 1
        else:
            num_special = len(agents_to_highlight)

        def animate(t):

            for ax, r, bounds  in zip(axes, r_list, bounds_list):
                ax.cla() # clear the previous image
                points_all = np.expand_dims(r[segment_indices[t]][:,other_ids,:], 1)
                segments_all = np.concatenate([points_all[:-1], points_all[1:]], axis = 1) # stack the beginning (t) and end (t+1) of each line-segment, across all agents
                segments_all_list = np.split(segments_all,num_other, axis=2) # one array of size (t_steps_back-1, 1, 2) per particle, in a list of such arrays
                lc1 = LineCollection(np.vstack(segments_all_list).squeeze(), colors = colors, linewidths=lwidths)
                collection = ax.add_collection(lc1)

                points_all = np.expand_dims(r[segment_indices[t]][:, agents_to_highlight, :],1)
                segments_all = np.concatenate([points_all[:-1], points_all[1:]], axis = 1) # stack the beginning (t) and end (t+1) of each line-segment, across all agents
                segments_all_list = np.split(segments_all, num_special, axis=2) # one array of size (t_steps_back-1, 1, 2) per particle, in a list of such arrays
                lc2 = LineCollection(np.vstack(segments_all_list).squeeze(), colors = highlight_colors, linewidths=highlight_lwidths[t])
                collection = ax.add_collection(lc2)

                if t <= (perturb_start_t + offset):
                    remaining_ts = (perturb_start_t + offset) - t
                    remaining_seconds = np.round(((perturb_start_t + offset) - t) * dt, decimals=2)
                    ax.set_title("Perturbation in %.2f seconds"%(remaining_seconds), fontsize=18)
                # ax.set_xlim(bounds[0])
                # ax.set_ylim(bounds[1])
                ax.set_xlim(bounds_list[1][0])
                ax.set_ylim(bounds_list[1][1])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                for spine in ax.spines.values():
                    spine.set_visible(False)

    anim = FuncAnimation(fig, animate, frames = t_indices, interval = 1, blit = False)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_name is None:
        save_name = f'perturbation_{N}_particles_{len(t_indices)}_ts_{timestr}.gif'
    
    full_save_path = os.path.join(save_dir, save_name)
    anim.save(full_save_path, writer='imagemagick', fps=fps)

    return full_save_path

def run_single_simulation(init_state, n_timesteps, genmodel, genproc, meta_params, returns = 'all', learning=False, learning_args=None):
    """ Wrapper function that runs a single realization of the active inference schooling simulation, with either learning or no-learning"""

    if learning:
        assert isinstance(learning_args, dict), "If you are using learning, must provide a dictionary containing dFdparam and param_mapping"
        # get single timestep function (learning version)
        single_timestep = make_single_timestep_fn(genproc, genmodel, learning_args['dFdparam'], learning_args['param_mapping'], meta_params)

        if returns == 'all':
            returns = ['pos', 'vel', 'mu', 'preparams', 'F']

        idx_to_return = []
        if 'pos' in returns:
            idx_to_return.append(0)
        if 'vel' in returns:
            idx_to_return.append(1)
        if 'mu' in returns:
            idx_to_return.append(2)
        if 'preparams' in returns:
            idx_to_return.append(3)
        if 'F' in returns:
            idx_to_return.append(4)

        # create a custom step function that will run the simulation as you want
        def step_fn(carry, t):
            pos_past, vel_past, mu_past, preparams_past = carry
            out = single_timestep(pos_past, vel_past, mu_past, preparams_past, t)
            pos, vel, mu, preparams, F = out
            return_states = tuple(out[idx] for idx in idx_to_return)
            return (pos, vel, mu, preparams), return_states
        _, history = lax.scan(step_fn, init_state, jnp.arange(n_timesteps))

    else:
        if learning_args is not None:
            print('Warning: although you have provided learning_args as a non-empty argument, it will be ignored because you also have set learning=False\n')
        # get single timestep function (no learning version)
        single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)

        if returns == 'all':
            returns = ['pos', 'vel', 'mu', 'F']
    
        idx_to_return = []
        if 'pos' in returns:
            idx_to_return.append(0)
        if 'vel' in returns:
            idx_to_return.append(1)
        if 'mu' in returns:
            idx_to_return.append(2)
        if 'preparams' in returns:
            print('You have asked to return preparams in the output, but learning has been turned off, so no preparams will be returned...\n')
        if 'F' in returns:
            idx_to_return.append(3)

        # create a custom step function that will run the simulation as you want
        def step_fn(carry, t):
            pos_past, vel_past, mu_past = carry
            out = single_timestep(pos_past, vel_past, mu_past, t)
            pos, vel, mu, F = out
            return_states = tuple(out[idx] for idx in idx_to_return)
            return (pos, vel, mu), return_states
        _, history = lax.scan(step_fn, init_state, jnp.arange(n_timesteps))

    history = history[0] if len(history) == 1 else history # if you're only returning one thing, just un-tuple already here so you don't return a 1-element tuple
    return history

def select_agents_to_perturb_empirical_metric(state, genmodel, N, dt, meta_params, real_key, n_agents = 1, real_T = 10, metric = 'cosine_angle', return_metric=False):
    """ 
    Selects a group of agents (`n_agents`) to perturb based on maximizing the change in the group heading (i.e. the direction of the group's center of mass)
    This change in the group heading is calculated as some metric (`metric`) that quantifies some difference between the group heading vector at the end of the perturbed realization 
    with the group heading at the beginning of the realization.
    """

    d_group_final = state[1].mean(axis=0)
    d_group_final = d_group_final / jnp.linalg.norm(d_group_final)

     # get initialization dictionary for the simulation
    init_dict = get_default_inits(N, real_T, dt)
    # initialize positions, velocities and other parameters of the generative process (e.g. noise realizations)
    _, _, genproc, _ = init_gen_process(real_key, init_dict)
    n_timesteps_perturb = len(genproc['t_axis'])

    if metric == 'cosine_angle':
        metric_fn = lambda x: compute_integrated_change_magnitude(d_group_final, x)
    elif metric == 'integrated_change_norm':
        metric_fn = lambda x: jnp.linalg.norm((x - d_group_final[None,:]), axis = 1).sum()

    out_metric_all_agents = compute_perturbability_score_empirical(state, genmodel, genproc, meta_params, n_timesteps_perturb, metric_fn)
   
    agent_ids = jnp.argsort(out_metric_all_agents)[::-1][:n_agents] # find first n_agents for whom `out_metric_all_agents` is largest

    # way to get the first `n_agents` agents for whom `out_metric_all_agents` is largest, that is compatible with dynamically shaped arrays (e.g. to allow JIT compilation)
    # agent_ids = lax.dynamic_slice_in_dim(jnp.argsort(out_metric_all_agents)[::-1], 0, n_agents) # find first n_agents for whom `out_metric_all_agents` is largest

    # if n_agents == 1, then we want to return a scalar, not a 1-element array
    agent_ids = agent_ids[0] if n_agents == 1 else agent_ids

    return (agent_ids, out_metric_all_agents) if return_metric else agent_ids
   
def compute_perturbability_score_empirical(state, genmodel, genproc, meta_params, n_timesteps_perturb, metric_fn):
    """
    Function that computes the perturbability score for each agent in the group, where the perturbability score is the change in the group heading vector,
    where the change in the heading vector is computed as some metric (`metric_fn`) that quantifies some difference between the group heading vector at the end of the perturbed realization
    """

    final_velocity = state[1]
    N = state[0].shape[0]

    def compute_perturbation_metric(agent_id):
        # get the velocity of the perturbed agent
        vel_to_rotate = final_velocity[agent_id,:]
        # rotate the velocity of the perturbed agent by 90 degrees
        perturbed_vel = final_velocity.at[agent_id,:].set(jnp.array([vel_to_rotate[1], -vel_to_rotate[0]]))
        perturbed_init_state = (state[0], perturbed_vel, state[2])
        pos_perturbed, vel_perturbed = run_single_simulation(perturbed_init_state, n_timesteps_perturb, genmodel, genproc, meta_params, returns = ['pos','vel'], learning=False)
        d_group_hist, _, _, _ = compute_Dgroup_and_rankings_vmapped(pos_perturbed, vel_perturbed)
        # susceptibility metric
        out_metric = metric_fn(d_group_hist)
        return out_metric
    
    out_metric_all_agents = vmap(compute_perturbation_metric)(jnp.arange(N))

    return out_metric_all_agents

def select_agents_to_perturb_empirical_metric_Couzin(state, N, n_timesteps_perturb, dt, noise_params, particle_params, n_agents = 1, real_T = 10, metric = 'cosine_angle', return_metric=False):
    """ 
    Selects a group of agents (`n_agents`) to perturb based on maximizing the change in the group heading (i.e. the direction of the group's center of mass)
    This change in the group heading is calculated as some metric (`metric`) that quantifies some difference between the group heading vector at the end of the perturbed realization 
    with the group heading at the beginning of the realization.
    """

    d_group_final = state[1].mean(axis=0)
    d_group_final = d_group_final / jnp.linalg.norm(d_group_final)

    if metric == 'cosine_angle':
        metric_fn = lambda x: compute_integrated_change_magnitude(d_group_final, x)
    elif metric == 'integrated_change_norm':
        metric_fn = lambda x: jnp.linalg.norm((x - d_group_final[None,:]), axis = 1).sum()

    out_metric_all_agents = compute_perturbability_score_empirical_Couzin(state, noise_params, particle_params, dt, n_timesteps_perturb, metric_fn)
   
    agent_ids = jnp.argsort(out_metric_all_agents)[::-1][:n_agents] # find first n_agents for whom `out_metric_all_agents` is largest

    # if n_agents == 1, then we want to return a scalar, not a 1-element array
    agent_ids = agent_ids[0] if n_agents == 1 else agent_ids

    return (agent_ids, out_metric_all_agents) if return_metric else agent_ids
   

def compute_perturbability_score_empirical_Couzin(state, noise_params, particle_params, dt, n_timesteps_perturb, metric_fn):
    """
    Function that computes the perturbability score for each agent in the group, where the perturbability score is the change in the group heading vector,
    where the change in the heading vector is computed as some metric (`metric_fn`) that quantifies some difference between the group heading vector at the end of the perturbed realization
    """

    final_velocity = state[1]
    N = state[0].shape[0]

    def compute_perturbation_metric(agent_id):
        # get the velocity of the perturbed agent
        vel_to_rotate = final_velocity[agent_id,:]
        # rotate the velocity of the perturbed agent by 90 degrees
        perturbed_vel = final_velocity.at[agent_id,:].set(jnp.array([vel_to_rotate[1], -vel_to_rotate[0]]))
        perturbed_init_state = (state[0], perturbed_vel)
        pos_perturbed, vel_perturbed = simulate_trajectories(perturbed_init_state, noise_params, particle_params, dt=dt, n_timesteps=n_timesteps_perturb)
        d_group_hist, _, _, _ = compute_Dgroup_and_rankings_vmapped(pos_perturbed, vel_perturbed)
        # susceptibility metric
        out_metric = metric_fn(d_group_hist)
        return out_metric
    
    out_metric_all_agents = vmap(compute_perturbation_metric)(jnp.arange(N))

    return out_metric_all_agents

def select_agents_to_perturb_vfe(state, genmodel, genproc, meta_params, n_agents=1, return_metric=False):
    """
    Function that returns the indices of agents that are candidates for perturbation, where the metric for 'perturbability' for agent_i
    is how much the group free energy changes as a function of agent_i's velocity or heading vector. The first `n_agents` with the highest perturbation
    scores are returned in an array called `agent_ids`
    """

    v_gradient_norm = compute_perturbability_score_vfe(state, genmodel, genproc, meta_params)
    agent_ids = jnp.argsort(v_gradient_norm)[::-1][:n_agents] # find first n_agents for whom this norm is largest

    agent_ids = agent_ids[0] if (n_agents==1) else agent_ids

    return (agent_ids, v_gradient_norm) if return_metric else agent_ids

def compute_perturbability_score_vfe(state, genmodel, genproc, meta_params):
    """ 
    Function that computes the VFE metric for 'perturbability' for all agents. The perturbability for agent_i
    is how much the group free energy changes as a function of agent_i's velocity or heading vector. 
    
    More detailed explanation:
    The perturbability score for each agent is measured by computing the gradient of the group-integrated free energy, a proxy for the group's "prediction error", 
    with respect to each agent's heading direction. This will return a shape (2,) gradient vector for each agent because velocities have an X and Y component. 
    Then to get a scalar score for each agent, we compute the norm of these gradient vectors, which gives a measure of the overall magnitude of change in group VFE 
    that would occur upon that agent's turning.
    """

    def compute_group_VFE_velocities(pos_t, vel_t, mu_t, t_idx):
        """ Function that computes the group free energy as a function of the positions, velocities, and beliefs of a group of agents """
        phi, all_dh_dr_self, empty_sectors_mask = get_observations_special(pos_t, vel_t, genproc, t_idx)
        infer_res, mu_traj = run_inference(phi, mu_t, empty_sectors_mask, genmodel, **meta_params['inference_params'])
        mu_next, _ = infer_res
        return compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel).sum()

    dFall_dvel= jit(grad(compute_group_VFE_velocities, argnums = 1)) # create function that computes derivatives of group VFE with respect to heading direction (in angles) of all agents

    pos, vel, mu, t = state
    dFall_dvel_eval = dFall_dvel(pos, vel, mu, t) # evaluate the gradients of the group's free energy with respect to each individual's velocity vector
    v_gradient_norm = jnp.linalg.norm(dFall_dvel_eval,axis=1) # take the norm of each of these gradient vectors, one norm computed per agent

    return v_gradient_norm

def compute_perturbability_score_sum_of_vfe(state, agent_indices, genmodel, genproc, meta_params, n_timesteps_perturb):
    """
    Function that computes the perturbability score for each agent in the group, where the perturbability score is the average
    variational free energy (averaged across timesteps and agents) of the perturbed realization after `n_timesteps_perturb` timesteps.
    """

    final_velocity = state[1]

    def compute_perturbation_metric(agent_id):
        # get the velocity of the perturbed agent
        vel_to_rotate = final_velocity[agent_id,:]
        # rotate the velocity of the perturbed agent by 90 degrees
        perturbed_vel = final_velocity.at[agent_id,:].set(jnp.array([vel_to_rotate[1], -vel_to_rotate[0]]))
        perturbed_init_state = (state[0], perturbed_vel, state[2])
        F_hist = run_single_simulation(perturbed_init_state, n_timesteps_perturb, genmodel, genproc, meta_params, returns = ['F'], learning=False)
        return F_hist.sum() / (n_timesteps_perturb * len(agent_indices)) # time average
    
    average_vfe_all_agents = vmap(compute_perturbation_metric)(agent_indices)

    return average_vfe_all_agents

def compute_vfe_cosangle_timecourse_per_agent(state, agent_indices, genmodel, genproc, meta_params, n_timesteps_perturb):

    final_velocity = state[1]

    def compute_perturbation_metric(agent_id):
        # get the velocity of the perturbed agent
        vel_to_rotate = final_velocity[agent_id,:]
        # rotate the velocity of the perturbed agent by 90 degrees
        perturbed_vel = final_velocity.at[agent_id,:].set(jnp.array([vel_to_rotate[1], -vel_to_rotate[0]]))
        perturbed_init_state = (state[0], perturbed_vel, state[2])
        pos_hist, vel_hist, F_hist = run_single_simulation(perturbed_init_state, n_timesteps_perturb, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'F'], learning=False)
        d_group_hist = vel_hist.mean(axis=1) 
        d_group_hist = d_group_hist / jnp.linalg.norm(d_group_hist, axis=1, keepdims=True) # normalize the group velocity over time
        cos_angles_over_time = -jnp.clip(jnp.dot(d_group_hist, d_group_hist[0]), -1.0, 1.0) 
        return pos_hist, F_hist, cos_angles_over_time
    
    return vmap(compute_perturbation_metric)(agent_indices)

def run_burnin_and_perturb_learning_old(real_key, N, dt, burn_in_time, perturb_time, init_state, n_agents_to_change, preparams, param_name, param_change_value, parameterization_mapping, genmodel, meta_params, learning_args):
    """
    Run a burn-in period and then a perturbation period, with learning enabled.
    """

    # get a new random key for the burn-in period
    burnin_key, perturb_key = random.split(real_key, num=2)

    #### Burn-in Period
    # get default initializations
    init_dict = get_default_inits(N, burn_in_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(burnin_key, init_dict)
    n_timesteps_burn_in = len(genproc['t_axis'])

    ####Run forward a copy of the simulation with learning enabled (burn-in period, before perturbation)
    pos_burnin, vel_burnin, mu_burnin, preparams_burnin = run_single_simulation(init_state + (preparams,), n_timesteps_burn_in, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'mu', 'preparams'], learning=True, learning_args=learning_args)

    # now select the agent to perturb after the burn-in period for the learning enabled condition
    burnin_final_state = (pos_burnin[-1], vel_burnin[-1], mu_burnin[-1])
    agent_id = select_agents_to_perturb_empirical_metric(burnin_final_state, genmodel, N, dt, meta_params, perturb_key, n_agents = n_agents_to_change, real_T = 5, metric = 'cosine_angle')

    #### Perturb Period
    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(perturb_key, init_dict) 
    n_timesteps_post_perturb = len(genproc['t_axis'])

    # take the last state of the burn-in period as the initial state for the perturbation period for the learning realization, using the learned parameters param_name) from the `preparams` dict
    last_param = preparams_burnin[param_name][-1]
    last_param_altered = last_param.at[agent_id].set(param_change_value) # set the `param_name` values of the selected agent(s) to the value you want to clamp them to
    preparams_postburnin = {param_name: last_param_altered}
    initial_learnable_params_postburnin = vmap(reparameterize, (0, None))(preparams_postburnin, parameterization_mapping)
    genmodel_postburnin = merge(genmodel, initial_learnable_params_postburnin)

    dFdparam_function_postburnin = make_dfdparams_fn(genmodel_postburnin, preparams_postburnin, parameterization_mapping, N)
    learning_args_postburnin = {'dFdparam': dFdparam_function_postburnin, 'param_mapping': parameterization_mapping}
    init_state_perturbed = burnin_final_state + (preparams_postburnin,)
    
    # fix the learning rate in the perturbed individual to 0.0 in order to disable learning in that individual
    learning_rates = meta_params['learning_params']['k_params']*jnp.ones(N)
    learning_rates = learning_rates.at[agent_id].set(0.0) # disable learning in the one individual selected for perturbation
    meta_params_onefrozen = initialize_meta_params(infer_lr = 0.1, 
                                    nsteps_infer = 1, 
                                    action_lr = 0.1, 
                                    nsteps_action = 1, 
                                    learning_lr=learning_rates,
                                    normalize_v = True
                                    )

    pos_perturb, vel_perturb, preparams_perturb = run_single_simulation(init_state_perturbed, n_timesteps_post_perturb, genmodel_postburnin, genproc, meta_params_onefrozen, returns = ['pos', 'vel', 'preparams'], learning=True, learning_args=learning_args_postburnin)
    
    # compute the turning magnitude for the perturbation period for the learning-enabled realization
    reference_velocity = init_state_perturbed[1]
    # perturb_angle_responses = compute_turning_magnitudes(reference_velocity, vel_perturb)

    # compute the history of cosine angles between the group velocity following the perturbation and the reference velocity
    d_group_hist, _, _, _ = compute_Dgroup_and_rankings_vmapped(pos_perturb, vel_perturb)
    d_group_ref = reference_velocity.mean(0) / jnp.linalg.norm(reference_velocity.mean(0))
    cos_angle_responses = -jnp.clip(jnp.dot(d_group_hist, d_group_ref), -1.0, 1.0)

    # compute the history of change-norms between the group velocity following the perturbation and the reference velocity
    change_norm_responses = jnp.linalg.norm((d_group_hist - d_group_ref[None,:]), axis = 1)

    # full preparams history (including burn-in and pertubarion periods)
    preparams_hist = jnp.concatenate((preparams_burnin[param_name], preparams_perturb[param_name]), axis = 0)
    
    # return preparams_hist, perturb_angle_responses, cos_angle_responses, change_norm_responses, agent_id
    return preparams_hist, cos_angle_responses, change_norm_responses, agent_id

def run_burnin_and_perturb_nolearning_old(real_key, N, dt, burn_in_time, perturb_time, init_state, n_agents_to_change, param_name, param_change_value, parameterization_mapping, genmodel, meta_params):
    """
    Run a burn-in period and then a perturbation period, with learning enabled.
    """

    # get a new random key for the burn-in period
    burnin_key, perturb_key = random.split(real_key, num=2)

    #### Burn-in Period
    # get default initializations
    init_dict = get_default_inits(N, burn_in_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(burnin_key, init_dict)
    n_timesteps_burn_in = len(genproc['t_axis'])

    # 1b.  Run forward a copy of the simulation with learning disabled (burn-in period, before perturbation)
    pos_burnin_noL, vel_burnin_noL, mu_burnin_noL = run_single_simulation(init_state, n_timesteps_burn_in, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'mu'], learning=False)
   
    # now select the agent to perturb after the burn-in period for the learning enabled condition
    burnin_final_state = (pos_burnin_noL[-1], vel_burnin_noL[-1], mu_burnin_noL[-1])
    agent_id = select_agents_to_perturb_empirical_metric(burnin_final_state, genmodel, N, dt, meta_params, perturb_key, n_agents = n_agents_to_change, real_T = 5, metric = 'cosine_angle')

    #### Perturb Period
    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(perturb_key, init_dict) 
    n_timesteps_post_perturb = len(genproc['t_axis'])

    # take the last state of the burn-in period as the initial state for the perturbation period for the no-learning realization
    last_param_altered = jnp.ones(N).at[agent_id].set(param_change_value) # set the `param_name` values of the selected agent(s) to the value you want to clamp them to
    initial_params_postburnin = vmap(reparameterize, (0, None))({param_name: last_param_altered}, parameterization_mapping)
    genmodel_postburnin = merge(genmodel, initial_params_postburnin)
    pos_perturb, vel_perturb = run_single_simulation(burnin_final_state, n_timesteps_post_perturb, genmodel_postburnin, genproc, meta_params, returns = ['pos','vel'], learning=False)
    
    # compute the turning magnitude for the perturbation period for the learning-disabled realization
    reference_velocity = burnin_final_state[1]
    # perturb_angle_responses = compute_turning_magnitudes(reference_velocity, vel_perturb)

    # compute the history of cosine angles between the group velocity following the perturbation and the reference velocity
    d_group_hist, _, _, _ = compute_Dgroup_and_rankings_vmapped(pos_perturb, vel_perturb)
    d_group_ref = reference_velocity.mean(0) / jnp.linalg.norm(reference_velocity.mean(0))
    cos_angle_responses = -jnp.clip(jnp.dot(d_group_hist, d_group_ref), -1.0, 1.0)

    # compute the history of change-norms between the group velocity following the perturbation and the reference velocity
    change_norm_responses = jnp.linalg.norm((d_group_hist - d_group_ref[None,:]), axis = 1)
    
    # return perturb_angle_responses, cos_angle_responses, change_norm_responses, agent_id
    return cos_angle_responses, change_norm_responses, agent_id

def run_burnin_and_perturb_learning(real_key, N, dt, burn_in_time, perturb_time, init_state, agent_id_vec, preparams, param_name, param_change_value, parameterization_mapping, genmodel, meta_params, learning_args, method='empirical'):
    """
    Run a burn-in period and then a perturbation period, with learning enabled. New version where instead of `n_agents_to_change` we simply
    have an `agent_id_vec` that specifies which agents to perturb, where the `agent_id_vec` is a vector of length N with 1's in the positions 
    of the agents to perturb and 0's in the positions of the agents to leave unchanged. Right now it is written where it's assumed that the first `n_agents_to_change`
    are the ones to perturb, but this can be changed by changing the `agent_id_vec` argument or via changing the nature of the `sorted_ids` variable in the function.
    Another change relative to the other version of this function, is that we're now no longer returning the `turning_angles` variables. 
    """

    # get a new random key for the burn-in period
    burnin_key, perturb_key = random.split(real_key, num=2)

    #### Burn-in Period
    # get default initializations
    init_dict = get_default_inits(N, burn_in_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(burnin_key, init_dict)
    n_timesteps_burn_in = len(genproc['t_axis'])

    ####Run forward a copy of the simulation with learning enabled (burn-in period, before perturbation)
    pos_burnin, vel_burnin, mu_burnin, preparams_burnin = run_single_simulation(init_state + (preparams,), n_timesteps_burn_in, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'mu', 'preparams'], learning=True, learning_args=learning_args)

    # now select the agent to perturb after the burn-in period for the learning enabled condition
    burnin_final_state = (pos_burnin[-1], vel_burnin[-1], mu_burnin[-1])

    if method == 'empirical':
        sorted_ids = select_agents_to_perturb_empirical_metric(burnin_final_state, genmodel, N, dt, meta_params, perturb_key, n_agents = N, real_T = 5, metric = 'cosine_angle')
    elif method == 'random':
        sorted_ids = random.choice(perturb_key, N, shape = (N,), replace = False)
    elif method == 'front_back':
        _, _, _, sorted_ids = compute_Dgroup_and_rankings_t(burnin_final_state[0], burnin_final_state[1])
        sorted_ids = sorted_ids[::-1]

    #### Perturb Period
    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(perturb_key, init_dict) 
    n_timesteps_post_perturb = len(genproc['t_axis'])

    # take the last state of the burn-in period as the initial state for the perturbation period for the learning realization, using the learned parameters param_name) from the `preparams` dict
    last_param = preparams_burnin[param_name][-1]
    last_param_vector = last_param[sorted_ids] * jnp.logical_not(agent_id_vec) + agent_id_vec * param_change_value # set the `param_name` values of the selected agent(s) to the value you want to clamp them to, while leaving the `param_name` values of the other agents unchanged
    last_param_altered = last_param.at[sorted_ids].set(last_param_vector)
    preparams_postburnin = {param_name: last_param_altered}
    initial_learnable_params_postburnin = vmap(reparameterize, (0, None))(preparams_postburnin, parameterization_mapping)
    genmodel_postburnin = merge(genmodel, initial_learnable_params_postburnin)

    dFdparam_function_postburnin = make_dfdparams_fn(genmodel_postburnin, preparams_postburnin, parameterization_mapping, N)
    learning_args_postburnin = {'dFdparam': dFdparam_function_postburnin, 'param_mapping': parameterization_mapping}
    init_state_perturbed = burnin_final_state + (preparams_postburnin,)
    
    # fix the learning rate in the perturbed individual(s) to 0.0 in order to disable learning in that(those) individual(s)
    learning_rates = meta_params['learning_params']['k_params']*jnp.ones(N)
    learning_rate_vector = learning_rates[sorted_ids] * jnp.logical_not(agent_id_vec) # set all learning rates to 0.0 except for the individuals selected for perturbation
    learning_rates_altered = learning_rates.at[sorted_ids].set(learning_rate_vector) # disable learning in the individuals selected for perturbation
    meta_params_onefrozen = initialize_meta_params(infer_lr = 0.1, 
                                    nsteps_infer = 1, 
                                    action_lr = 0.1, 
                                    nsteps_action = 1, 
                                    learning_lr=learning_rates_altered,
                                    normalize_v = True
                                    )

    pos_perturb, vel_perturb, preparams_perturb = run_single_simulation(init_state_perturbed, n_timesteps_post_perturb, genmodel_postburnin, genproc, meta_params_onefrozen, returns = ['pos', 'vel', 'preparams'], learning=True, learning_args=learning_args_postburnin)
    
    # compute the history of cosine angles between the group velocity following the perturbation and the reference velocity
    reference_velocity = init_state_perturbed[1][sorted_ids] * jnp.logical_not(agent_id_vec)[...,None]
    d_group_ref = reference_velocity.sum(axis=0) / (N - agent_id_vec.sum())
    d_group_ref = d_group_ref / jnp.linalg.norm(d_group_ref)

    # compute the heading direction of the group
    vel_filtered = vel_perturb[:,sorted_ids,:] * jnp.logical_not(agent_id_vec)[None,:,None]
    d_group_hist = vel_filtered.sum(axis=1) / (N - agent_id_vec.sum()) # compute the group velocity, ignoring the perturbed agent(s)
    # normalize to unit vector
    d_group_hist = d_group_hist / jnp.linalg.norm(d_group_hist, axis=1, keepdims=True)
    
    cos_angle_responses = -jnp.clip(jnp.dot(d_group_hist, d_group_ref), -1.0, 1.0)
    
    # compute the history of change-norms between the group velocity following the perturbation and the reference velocity
    change_norm_responses = jnp.linalg.norm((d_group_hist - d_group_ref[None,:]), axis = 1)

    # full preparams history (including burn-in and pertubarion periods)
    preparams_hist = jnp.concatenate((preparams_burnin[param_name], preparams_perturb[param_name]), axis = 0)
    
    return preparams_hist, cos_angle_responses, change_norm_responses, sorted_ids

def run_burnin_and_perturb_nolearning(real_key, N, dt, burn_in_time, perturb_time, init_state, agent_id_vec, param_name, param_change_value, parameterization_mapping, genmodel, meta_params, method='empirical'):
    """
    Run a burn-in period and then a perturbation period, with learning disabled. New version where instead of `n_agents_to_change` we simply
    have an `agent_id_vec` that specifies which agents to perturb, where the `agent_id_vec` is a vector of length N with 1's in the positions 
    of the agents to perturb and 0's in the positions of the agents to leave unchanged. Right now it is written where it's assumed that the first `n_agents_to_change`
    are the ones to perturb, but this can be changed by changing the `agent_id_vec` argument or via changing the nature of the `sorted_ids` variable in the function.
    Another change relative to the other version of this function, is that we're now no longer returning the `turning_angles` variables. 
    """

    # get a new random key for the burn-in period
    burnin_key, perturb_key = random.split(real_key, num=2)

    #### Burn-in Period
    # get default initializations
    init_dict = get_default_inits(N, burn_in_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(burnin_key, init_dict)
    n_timesteps_burn_in = len(genproc['t_axis'])

    # 1b.  Run forward a copy of the simulation with learning disabled (burn-in period, before perturbation)
    pos_burnin_noL, vel_burnin_noL, mu_burnin_noL = run_single_simulation(init_state, n_timesteps_burn_in, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'mu'], learning=False)
   
    # now select the agent to perturb after the burn-in period for the learning enabled condition
    burnin_final_state = (pos_burnin_noL[-1], vel_burnin_noL[-1], mu_burnin_noL[-1])

    if method == 'empirical':
        # select the agents to perturb based on the empirical metric
        sorted_ids = select_agents_to_perturb_empirical_metric(burnin_final_state, genmodel, N, dt, meta_params, perturb_key, n_agents = N, real_T = 5, metric = 'cosine_angle')
    elif method == 'random':
        sorted_ids = random.choice(perturb_key, N, shape = (N,), replace = False)
    elif method == 'front_back':
        _, _, _, sorted_ids = compute_Dgroup_and_rankings_t(burnin_final_state[0], burnin_final_state[1])
        sorted_ids = sorted_ids[::-1]

    #### Perturb Period
    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(perturb_key, init_dict) 
    n_timesteps_post_perturb = len(genproc['t_axis'])

    # take the last state of the burn-in period as the initial state for the perturbation period for the no-learning realization
    last_param = jnp.ones(N) # default value of `s_z` is 1.0 @TODO: This should be changed to confrom to its true value in `genmodel` at some point
    last_param_vector = last_param[sorted_ids] * jnp.logical_not(agent_id_vec) + agent_id_vec * param_change_value # set the `param_name` values of the selected agent(s) to the value you want to clamp them to, while leaving the `param_name` values of the other agents unchanged
    last_param_altered = last_param.at[sorted_ids].set(last_param_vector)    
    initial_params_postburnin = vmap(reparameterize, (0, None))({param_name: last_param_altered}, parameterization_mapping)
    genmodel_postburnin = merge(genmodel, initial_params_postburnin)

    pos_perturb, vel_perturb = run_single_simulation(burnin_final_state, n_timesteps_post_perturb, genmodel_postburnin, genproc, meta_params, returns = ['pos','vel'], learning=False)
    
    # compute the history of cosine angles between the group velocity following the perturbation and the reference velocity
    reference_velocity = init_state_perturbed[1][sorted_ids] * jnp.logical_not(agent_id_vec)[...,None]
    d_group_ref = reference_velocity.sum(axis=0) / (N - agent_id_vec.sum())
    d_group_ref = d_group_ref / jnp.linalg.norm(d_group_ref)

    # compute the heading direction of the group
    vel_filtered = vel_perturb[:,sorted_ids,:] * jnp.logical_not(agent_id_vec)[None,:,None]
    d_group_hist = vel_filtered.sum(axis=1) / (N - agent_id_vec.sum()) # compute the group velocity, ignoring the perturbed agent(s)
    # normalize to unit vector
    d_group_hist = d_group_hist / jnp.linalg.norm(d_group_hist, axis=1, keepdims=True)
    
    cos_angle_responses = -jnp.clip(jnp.dot(d_group_hist, d_group_ref), -1.0, 1.0)

    # compute the history of change-norms between the group velocity following the perturbation and the reference velocity
    change_norm_responses = jnp.linalg.norm((d_group_hist - d_group_ref[None,:]), axis = 1)
    
    return cos_angle_responses, change_norm_responses, sorted_ids

def run_burnin_and_perturb_vel_learning(real_key, N, dt, burn_in_time, perturb_time, init_state, agent_id_vec, preparams, param_name, parameterization_mapping, genmodel, meta_params, learning_args, method = 'empirical'):
    """
    Run a burn-in period and then a perturbation period, with learning enabled. New version where instead of `n_agents_to_change` we simply
    have an `agent_id_vec` that specifies which agents to perturb, where the `agent_id_vec` is a vector of length N with 1's in the positions 
    of the agents to perturb and 0's in the positions of the agents to leave unchanged. Right now it is written where it's assumed that the first `n_agents_to_change`
    are the ones to perturb, but this can be changed by changing the `agent_id_vec` argument or via changing the nature of the `sorted_ids` variable in the function.
    Another change relative to the other version of this function, is that we're now no longer returning the `turning_angles` variables. 
    """

    # get a new random key for the burn-in period
    burnin_key, perturb_key = random.split(real_key, num=2)

    #### Burn-in Period
    # get default initializations
    init_dict = get_default_inits(N, burn_in_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(burnin_key, init_dict)
    n_timesteps_burn_in = len(genproc['t_axis'])

    ####Run forward a copy of the simulation with learning enabled (burn-in period, before perturbation)
    pos_burnin, vel_burnin, mu_burnin, preparams_burnin = run_single_simulation(init_state + (preparams,), n_timesteps_burn_in, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'mu', 'preparams'], learning=True, learning_args=learning_args)

    # now select the agent to perturb after the burn-in period for the learning enabled condition
    burnin_final_state = (pos_burnin[-1], vel_burnin[-1], mu_burnin[-1])

    if method == 'empirical':
        sorted_ids = select_agents_to_perturb_empirical_metric(burnin_final_state, genmodel, N, dt, meta_params, perturb_key, n_agents = N, real_T = 5, metric = 'cosine_angle')
    elif method == 'random':
        sorted_ids = random.choice(perturb_key, N, shape = (N,), replace = False)
    elif method == 'front_back':
        _, _, _, sorted_ids = compute_Dgroup_and_rankings_t(burnin_final_state[0], burnin_final_state[1])
        sorted_ids = sorted_ids[::-1]

    #### Perturb Period
    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(perturb_key, init_dict) 
    n_timesteps_post_perturb = len(genproc['t_axis'])

    # take the last state of the burn-in period as the initial state for the perturbation period for the learning realization, using the final state of the velocity vectors of the perturbed agents
    last_vel = burnin_final_state[1]
    last_vel_rotated = last_vel[sorted_ids] @ jnp.array([[0,1],[-1,0]]) # rotate all velocity vectors by 90 degrees
    last_vel_rotated = last_vel[sorted_ids] * jnp.logical_not(agent_id_vec)[...,None] + last_vel_rotated * agent_id_vec[...,None] # set the velocity vectors of the selected agent(s) to the value you want to clamp them to, while leaving the velocity vectors of the other agents unchanged
    last_vel_rotated = last_vel.at[sorted_ids].set(last_vel_rotated) # resort the velocity vectors

    preparams_postburnin = {param_name: preparams_burnin[param_name][-1]}
    initial_learnable_params_postburnin = vmap(reparameterize, (0, None))(preparams_postburnin, parameterization_mapping)
    genmodel_postburnin = merge(genmodel, initial_learnable_params_postburnin)

    dFdparam_function_postburnin = make_dfdparams_fn(genmodel_postburnin, preparams_postburnin, parameterization_mapping, N)
    learning_args_postburnin = {'dFdparam': dFdparam_function_postburnin, 'param_mapping': parameterization_mapping}
    init_state_perturbed = (burnin_final_state[0], last_vel_rotated, burnin_final_state[2], preparams_postburnin)

    pos_perturb, vel_perturb, preparams_perturb = run_single_simulation(init_state_perturbed, n_timesteps_post_perturb, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'preparams'], learning=True, learning_args=learning_args_postburnin)
    
    # compute the history of cosine angles between the group velocity following the perturbation and the reference velocity
    reference_velocity = init_state_perturbed[1][sorted_ids] * jnp.logical_not(agent_id_vec)[...,None]
    d_group_ref = reference_velocity.sum(axis=0) / (N - agent_id_vec.sum())
    d_group_ref = d_group_ref / jnp.linalg.norm(d_group_ref)

    # compute the heading direction of the group
    vel_filtered = vel_perturb[:,sorted_ids,:] * jnp.logical_not(agent_id_vec)[None,:,None]
    d_group_hist = vel_filtered.sum(axis=1) / (N - agent_id_vec.sum()) # compute the group velocity, ignoring the perturbed agent(s)
    # normalize to unit vector
    d_group_hist = d_group_hist / jnp.linalg.norm(d_group_hist, axis=1, keepdims=True)
    
    cos_angle_responses = -jnp.clip(jnp.dot(d_group_hist, d_group_ref), -1.0, 1.0)

    # compute the history of change-norms between the group velocity following the perturbation and the reference velocity
    change_norm_responses = jnp.linalg.norm((d_group_hist - d_group_ref[None,:]), axis = 1)

    # full preparams history (including burn-in and pertubarion periods)
    preparams_hist = jnp.concatenate((preparams_burnin[param_name], preparams_perturb[param_name]), axis = 0)
    
    return preparams_hist, cos_angle_responses, change_norm_responses, sorted_ids

def run_burnin_and_perturb_vel_nolearning(real_key, N, dt, burn_in_time, perturb_time, init_state, agent_id_vec, param_name, parameterization_mapping, genmodel, meta_params, method='empirical'):
    """
    Run a burn-in period and then a perturbation period, with learning disabled. New version where instead of `n_agents_to_change` we simply
    have an `agent_id_vec` that specifies which agents to perturb, where the `agent_id_vec` is a vector of length N with 1's in the positions 
    of the agents to perturb and 0's in the positions of the agents to leave unchanged. Right now it is written where it's assumed that the first `n_agents_to_change`
    are the ones to perturb, but this can be changed by changing the `agent_id_vec` argument or via changing the nature of the `sorted_ids` variable in the function.
    Another change relative to the other version of this function, is that we're now no longer returning the `turning_angles` variables. 
    """

    # get a new random key for the burn-in period
    burnin_key, perturb_key = random.split(real_key, num=2)

    #### Burn-in Period
    # get default initializations
    init_dict = get_default_inits(N, burn_in_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(burnin_key, init_dict)
    n_timesteps_burn_in = len(genproc['t_axis'])

    # 1b.  Run forward a copy of the simulation with learning disabled (burn-in period, before perturbation)
    pos_burnin_noL, vel_burnin_noL, mu_burnin_noL = run_single_simulation(init_state, n_timesteps_burn_in, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'mu'], learning=False)
   
    # now select the agent to perturb after the burn-in period for the learning enabled condition
    burnin_final_state = (pos_burnin_noL[-1], vel_burnin_noL[-1], mu_burnin_noL[-1])

    if method == 'empirical':
        sorted_ids = select_agents_to_perturb_empirical_metric(burnin_final_state, genmodel, N, dt, meta_params, perturb_key, n_agents = N, real_T = 5, metric = 'cosine_angle')
    elif method == 'random':
        sorted_ids = random.choice(perturb_key, N, shape = (N,), replace = False)
    elif method == 'front_back':
        _, _, _, sorted_ids = compute_Dgroup_and_rankings_t(burnin_final_state[0], burnin_final_state[1])
        sorted_ids = sorted_ids[::-1]

    #### Perturb Period
    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(perturb_key, init_dict) 
    n_timesteps_post_perturb = len(genproc['t_axis'])

    # take the last state of the burn-in period as the initial state for the perturbation period for the learning realization, using the final state of the velocity vectors of the perturbed agents
    last_vel = burnin_final_state[1]
    last_vel_rotated = last_vel[sorted_ids] @ jnp.array([[0,1],[-1,0]]) # rotate all velocity vectors by 90 degrees
    last_vel_rotated = last_vel[sorted_ids] * jnp.logical_not(agent_id_vec)[...,None] + last_vel_rotated * agent_id_vec[...,None] # set the velocity vectors of the selected agent(s) to the value you want to clamp them to, while leaving the velocity vectors of the other agents unchanged
    last_vel_rotated = last_vel.at[sorted_ids].set(last_vel_rotated) # resort the velocity vectors

    init_state_perturbed = (burnin_final_state[0], last_vel_rotated, burnin_final_state[2])

    pos_perturb, vel_perturb = run_single_simulation(init_state_perturbed, n_timesteps_post_perturb, genmodel, genproc, meta_params, returns = ['pos','vel'], learning=False)
    
    # compute the history of cosine angles between the group velocity following the perturbation and the reference velocity
    reference_velocity = init_state_perturbed[1][sorted_ids] * jnp.logical_not(agent_id_vec)[...,None]
    d_group_ref = reference_velocity.sum(axis=0) / (N - agent_id_vec.sum())
    d_group_ref = d_group_ref / jnp.linalg.norm(d_group_ref)

    # compute the heading direction of the group
    vel_filtered = vel_perturb[:,sorted_ids,:] * jnp.logical_not(agent_id_vec)[None,:,None]
    d_group_hist = vel_filtered.sum(axis=1) / (N - agent_id_vec.sum()) # compute the group velocity, ignoring the perturbed agent(s)
    # normalize to unit vector
    d_group_hist = d_group_hist / jnp.linalg.norm(d_group_hist, axis=1, keepdims=True)
    
    cos_angle_responses = -jnp.clip(jnp.dot(d_group_hist, d_group_ref), -1.0, 1.0)

    # compute the history of change-norms between the group velocity following the perturbation and the reference velocity
    change_norm_responses = jnp.linalg.norm((d_group_hist - d_group_ref[None,:]), axis = 1)
    
    return cos_angle_responses, change_norm_responses, sorted_ids

def run_burnin_and_perturb_noise_learning(real_key, N, dt, burn_in_time, perturb_time, noise_structure, init_state, agent_id_vec, preparams, param_name, parameterization_mapping, genmodel, meta_params, learning_args, method = 'empirical'):
    """
    Run a burn-in period and then a perturbation period, with learning enabled. Another change relative to the other version of this function, is that we're now no longer returning the `turning_angles` variables. 
    This version is different because we only perturb one individual and now we're doing it with structured noise.
    """

    # get a new random key for the burn-in period
    burnin_key, perturb_key = random.split(real_key, num=2)

    #### Burn-in Period
    # get default initializations
    init_dict = get_default_inits(N, burn_in_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(burnin_key, init_dict)
    n_timesteps_burn_in = len(genproc['t_axis'])

    ####Run forward a copy of the simulation with learning enabled (burn-in period, before perturbation)
    pos_burnin, vel_burnin, mu_burnin, preparams_burnin = run_single_simulation(init_state + (preparams,), n_timesteps_burn_in, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'mu', 'preparams'], learning=True, learning_args=learning_args)

    # now select the agent to perturb after the burn-in period for the learning enabled condition
    burnin_final_state = (pos_burnin[-1], vel_burnin[-1], mu_burnin[-1])

    if method == 'empirical':
        sorted_ids = select_agents_to_perturb_empirical_metric(burnin_final_state, genmodel, N, dt, meta_params, perturb_key, n_agents = N, real_T = 5, metric = 'cosine_angle')
    elif method == 'random':
        sorted_ids = random.choice(perturb_key, N, shape = (N,), replace = False)
    elif method == 'front_back':
        _, _, _, sorted_ids = compute_Dgroup_and_rankings_t(burnin_final_state[0], burnin_final_state[1])
        sorted_ids = sorted_ids[::-1]

    #### Perturb Period
    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(perturb_key, init_dict) 
    n_timesteps_post_perturb = len(genproc['t_axis'])

    t_end = noise_structure.shape[0]
    sensory_noise_all = genproc['sensory_noise'][:t_end,1,:,:]
    noise_perturbed_all = sensory_noise_all + noise_structure[...,None]
    noise_perturbed = sensory_noise_all[:,:,sorted_ids] * jnp.logical_not(agent_id_vec)[None,None,...] + noise_perturbed_all[:,:,sorted_ids] * agent_id_vec[None,None,...] # set the velocity vectors of the selected agent(s) to the value you want to clamp them to, while leaving the velocity vectors of the other agents unchanged
    genproc['sensory_noise'] = genproc['sensory_noise'].at[:t_end,1,:,:].set(noise_perturbed[:,:,jnp.argsort(sorted_ids)])
    
    preparams_postburnin = {param_name: preparams_burnin[param_name][-1]}
    initial_learnable_params_postburnin = vmap(reparameterize, (0, None))(preparams_postburnin, parameterization_mapping)
    genmodel_postburnin = merge(genmodel, initial_learnable_params_postburnin)

    dFdparam_function_postburnin = make_dfdparams_fn(genmodel_postburnin, preparams_postburnin, parameterization_mapping, N)
    learning_args_postburnin = {'dFdparam': dFdparam_function_postburnin, 'param_mapping': parameterization_mapping}
    init_state_perturbed = burnin_final_state + (preparams_postburnin,)

    pos_perturb, vel_perturb, preparams_perturb = run_single_simulation(init_state_perturbed, n_timesteps_post_perturb, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'preparams'], learning=True, learning_args=learning_args_postburnin)
    
    # compute the history of cosine angles between the group velocity following the perturbation and the reference velocity
    reference_velocity = init_state_perturbed[1][sorted_ids] * jnp.logical_not(agent_id_vec)[...,None]
    d_group_ref = reference_velocity.sum(axis=0) / (N - agent_id_vec.sum())
    d_group_ref = d_group_ref / jnp.linalg.norm(d_group_ref)

    # compute the heading direction of the group
    vel_filtered = vel_perturb[:,sorted_ids,:] * jnp.logical_not(agent_id_vec)[None,:,None]
    d_group_hist = vel_filtered.sum(axis=1) / (N - agent_id_vec.sum()) # compute the group velocity, ignoring the perturbed agent(s)
    # normalize to unit vector
    d_group_hist = d_group_hist / jnp.linalg.norm(d_group_hist, axis=1, keepdims=True)
    
    cos_angle_responses = -jnp.clip(jnp.dot(d_group_hist, d_group_ref), -1.0, 1.0)

    # compute the history of change-norms between the group velocity following the perturbation and the reference velocity
    change_norm_responses = jnp.linalg.norm((d_group_hist - d_group_ref[None,:]), axis = 1)

    # full preparams history (including burn-in and pertubarion periods)
    preparams_hist = jnp.concatenate((preparams_burnin[param_name], preparams_perturb[param_name]), axis = 0)
    
    return preparams_hist, cos_angle_responses, change_norm_responses, sorted_ids

def run_burnin_and_perturb_noise_nolearning(real_key, N, dt, burn_in_time, perturb_time, noise_structure, init_state, agent_id_vec, param_name, parameterization_mapping, genmodel, meta_params, method='empirical'):
    """
    Run a burn-in period and then a perturbation period, with learning disabled. New version where instead of `n_agents_to_change` we simply
    have an `agent_id_vec` that specifies which agents to perturb, where the `agent_id_vec` is a vector of length N with 1's in the positions 
    of the agents to perturb and 0's in the positions of the agents to leave unchanged. Right now it is written where it's assumed that the first `n_agents_to_change`
    are the ones to perturb, but this can be changed by changing the `agent_id_vec` argument or via changing the nature of the `sorted_ids` variable in the function.
    Another change relative to the other version of this function, is that we're now no longer returning the `turning_angles` variables. 
    """

    # get a new random key for the burn-in period
    burnin_key, perturb_key = random.split(real_key, num=2)

    #### Burn-in Period
    # get default initializations
    init_dict = get_default_inits(N, burn_in_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(burnin_key, init_dict)
    n_timesteps_burn_in = len(genproc['t_axis'])

    # 1b.  Run forward a copy of the simulation with learning disabled (burn-in period, before perturbation)
    pos_burnin_noL, vel_burnin_noL, mu_burnin_noL = run_single_simulation(init_state, n_timesteps_burn_in, genmodel, genproc, meta_params, returns = ['pos', 'vel', 'mu'], learning=False)
   
    # now select the agent to perturb after the burn-in period for the learning enabled condition
    burnin_final_state = (pos_burnin_noL[-1], vel_burnin_noL[-1], mu_burnin_noL[-1])

    if method == 'empirical':
        sorted_ids = select_agents_to_perturb_empirical_metric(burnin_final_state, genmodel, N, dt, meta_params, perturb_key, n_agents = N, real_T = 5, metric = 'cosine_angle')
    elif method == 'random':
        sorted_ids = random.choice(perturb_key, N, shape = (N,), replace = False)
    elif method == 'front_back':
        _, _, _, sorted_ids = compute_Dgroup_and_rankings_t(burnin_final_state[0], burnin_final_state[1])
        sorted_ids = sorted_ids[::-1]

    #### Perturb Period
    # get default initializations
    init_dict = get_default_inits(N, perturb_time, dt)
    # initialize generative process
    _, _, genproc, _ = init_gen_process(perturb_key, init_dict) 
    n_timesteps_post_perturb = len(genproc['t_axis'])

    t_end = noise_structure.shape[0]
    sensory_noise_all = genproc['sensory_noise'][:t_end,1,:,:]
    noise_perturbed_all = sensory_noise_all + noise_structure[...,None]
    noise_perturbed = sensory_noise_all[:,:,sorted_ids] * jnp.logical_not(agent_id_vec)[None,None,...] + noise_perturbed_all[:,:,sorted_ids] * agent_id_vec[None,None,...] # set the velocity vectors of the selected agent(s) to the value you want to clamp them to, while leaving the velocity vectors of the other agents unchanged
    genproc['sensory_noise'] = genproc['sensory_noise'].at[:t_end,1,:,:].set(noise_perturbed[:,:,jnp.argsort(sorted_ids)])
    
    init_state_perturbed = (burnin_final_state[0], burnin_final_state[1], burnin_final_state[2])

    pos_perturb, vel_perturb = run_single_simulation(init_state_perturbed, n_timesteps_post_perturb, genmodel, genproc, meta_params, returns = ['pos','vel'], learning=False)
    
    # compute the history of cosine angles between the group velocity following the perturbation and the reference velocity
    reference_velocity = init_state_perturbed[1][sorted_ids] * jnp.logical_not(agent_id_vec)[...,None]
    d_group_ref = reference_velocity.sum(axis=0) / (N - agent_id_vec.sum())
    d_group_ref = d_group_ref / jnp.linalg.norm(d_group_ref)

    # compute the heading direction of the group
    vel_filtered = vel_perturb[:,sorted_ids,:] * jnp.logical_not(agent_id_vec)[None,:,None]
    d_group_hist = vel_filtered.sum(axis=1) / (N - agent_id_vec.sum()) # compute the group velocity, ignoring the perturbed agent(s)
    # normalize to unit vector
    d_group_hist = d_group_hist / jnp.linalg.norm(d_group_hist, axis=1, keepdims=True)
    
    cos_angle_responses = -jnp.clip(jnp.dot(d_group_hist, d_group_ref), -1.0, 1.0)

    # compute the history of change-norms between the group velocity following the perturbation and the reference velocity
    change_norm_responses = jnp.linalg.norm((d_group_hist - d_group_ref[None,:]), axis = 1)
    
    return cos_angle_responses, change_norm_responses, sorted_ids

