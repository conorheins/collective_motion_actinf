
import os
import argparse
import jax
from jax import random, lax, vmap, jit, grad
from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
import pickle
from funcy import merge
import scipy.signal

from utils import make_single_timestep_fn, make_single_timestep_fn_nolearning, initialize_meta_params, get_default_inits, animate_trajectory, animate_trajectories_compare
from learning import reparameterize, make_dfdparams_fn
from genprocess import init_gen_process, get_observations_special, compute_turning_magnitudes
from genmodel import init_genmodel, compute_vfe_vectorized
from inference import run_inference


def parameterize_Pi_2do(s_z):
    """
    Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
    """
    Pi_z_temporal = jnp.diag(jnp.array([1.0, 2 * s_z**2]))
    return jnp.kron(Pi_z_temporal, jnp.eye(4))

def smooth_trajectories(trajectories, window_size = 5):

    smoothed_traj = np.zeros_like(trajectories)
    conv_window = (np.ones(window_size) / window_size).reshape(-1,1)
    for n in range(trajectories.shape[1]):
        smoothed_traj[:,n,:] = scipy.signal.convolve2d(trajectories[:,n,:], conv_window, mode='full')[:-(window_size-1)] # chop off the end
    return smoothed_traj

def run(init_key_num = 1, # initial key number for the JAX PRNG seed
        N = 30, # number of individuals
        action_variance=0.01, # variance of actions
        T = 50, # duration of simulation in seconds
        perturb_length = 10, #  duration of time (in seconds) of simulation recorded following the perturbation
        perturb_value = -5., # scalar value of the perturbation
        perturb_duration = 10, # number of timesteps that the change to the observations lasts (the perturbation intervention itself)
        n_pre = 1000, # number of timesteps to show trajectories pre-perturbation
        start_t = 0, # time from the beginning of the animation block to start
        end_t = 1500, # time from beginning of the animation to end
        skip = 1, # how many timesteps to skip in between frames
        t_steps_back = 5, # how long each trajectory should be shown back in time before vanishing
        smooth_flag = False, # boolean for whether to smooth the trajectory or not
        window_size = 5, # size of convolutional window for smoothing
        fps = 20 # frames per second to render the gif
        ):

        init_key = random.PRNGKey(init_key_num)

        dt, D = 0.01, 2 

        n_timesteps = len(jnp.arange(start=0, stop=T, step=dt))

        init_dict = get_default_inits(N, T, dt)
        init_dict['z_action'] = action_variance

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
        pre_perturb_pos, pre_perturb_vel, pre_perturb_mu = history[0], history[1], history[2]

        def compute_group_VFE_velocities(pos_t, vel_t, mu_t, t_idx):
            """ Function that computes the group free energy as a function of the positions, velocities, and beliefs of a group of agents """
            phi, all_dh_dr_self, empty_sectors_mask = get_observations_special(pos_t, vel_t, genproc, t_idx)
            infer_res, mu_traj = run_inference(phi, mu_t, empty_sectors_mask, genmodel, **meta_params['inference_params'])
            mu_next, _ = infer_res
            return compute_vfe_vectorized(mu_next, phi, empty_sectors_mask, genmodel).sum()

        dFall_dvel= jit(grad(compute_group_VFE_velocities, argnums = 1)) # create function that computes derivatives of group VFE with respect to heading direction (in angles) of all agents

        dFall_dvel_eval = dFall_dvel(pre_perturb_pos[-1], pre_perturb_vel[-1], pre_perturb_mu[-1], n_timesteps-1)
        v_gradient_norm = jnp.linalg.norm(dFall_dvel_eval,axis=1) # take the norm of each of these gradient vectors, one norm computed per agent
        agent_id = jnp.argmax(v_gradient_norm) # find the agent for whom this norm is largest
        n_timesteps_perturb = len(jnp.arange(start=0, stop=perturb_length, step=dt))
        
        """ Now set up the perturbation process, no learning"""
        # get default initializations
        init_dict = get_default_inits(N, perturb_length, dt)

        # initialize generative model, generative process, and meta parameters related to learning and inference
        _, _, genproc, _ = init_gen_process(new_key, init_dict)
        genmodel = init_genmodel(init_dict)

        # introduce a perturbation by incrementing the velocity observations of the chosen agent by a fixed value for a fixed duration of time 
        noise_tensor = genproc['sensory_noise']
        genproc['sensory_noise'] = noise_tensor.at[:perturb_duration,1,:,agent_id].set(noise_tensor[:perturb_duration,1,:,agent_id] + perturb_value)

        # initialize the state to be the same for all perturbations, which is the last point in history of the original simulation
        init_state = (pre_perturb_pos[-1], pre_perturb_vel[-1], pre_perturb_mu[-1])
        
        single_timestep = make_single_timestep_fn_nolearning(genproc, genmodel, meta_params)
        # create a custom step function that will run the simulation as you want
        def step_fn(carry, t):
            pos_past, vel_past, mu_past = carry
            pos, vel, mu, F = single_timestep(pos_past, vel_past, mu_past, t)
            return (pos, vel, mu), pos
        _, perturb_pos = lax.scan(step_fn, init_state, jnp.arange(n_timesteps_perturb))

        no_learning_total_pos = jnp.concatenate([pre_perturb_pos[-n_pre:], perturb_pos], axis = 0)

        # final_save_name = animate_trajectory(no_learning_total_pos, start_t=start_t, end_t=1500, skip=skip, t_steps_back=t_steps_back)

        """ Now set up the perturbation process, with learning """
        
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
            return (pos, vel, mu, preparams), pos
        _, perturb_pos = lax.scan(step_fn, init_state + (preparams,),  jnp.arange(n_timesteps_perturb))

        learning_total_pos = jnp.concatenate([pre_perturb_pos[-n_pre:], perturb_pos], axis = 0)

        both_trajectories = [no_learning_total_pos, learning_total_pos]

        if smooth_flag:
            both_trajectories = [smooth_trajectories(np.array(r), window_size=window_size) for r in both_trajectories]
        else:
            both_trajectories = [np.array(r) for r in both_trajectories]

        if agent_id.size == 1:
            agent_id = int(agent_id)
        
        # print(both_trajectories[0].shape)
        final_save_name = animate_trajectories_compare(both_trajectories, dt=dt, start_t=start_t, end_t=end_t, perturb_start_t=n_pre, skip=skip, t_steps_back=t_steps_back, agents_to_highlight=agent_id, fps=fps)
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
    parser.add_argument('--action_variance', '-zha', type = float,
                help = "Variance of action",
                dest = "action_variance", default=0.01)
    parser.add_argument('--T', '-T', type = int,
                help = "Duration of simulation (in seconds)",
                dest = "T", default=50)
    parser.add_argument('--start_t', '-st', type = int,
                help = "Start timestep (in absolute time) of animation",
                dest = "start_t", default=0)
    parser.add_argument('--end_t', '-et', type = int,
                help = "End timestep (in absolute time) of animation",
                dest = "end_t", default=1500)
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
    args = parser.parse_args()
    run(**vars(args))

# fig, ax = plt.subplots(figsize=(10,8))
# for i in range(N):
#     ax.plot(r[:,i,0], r[:,i,1])

# plt.show()


# fig, ax= plt.subplots(figsize=(6,6))
 
# start_t, end_t, skip = n_timesteps - 500, n_timesteps-1, 10
# qv = ax.quiver(r[start_t,:,0], r[start_t,:,1], v[start_t,:,0], v[start_t,:,1], theta[start_t,:], clim=[-np.pi, np.pi])


# x_bounds = [np.min(r[start_t:(end_t+1),:,0]), np.max(r[start_t:(end_t+1),:,0])]
# y_bounds = [np.min(r[start_t:(end_t+1),:,1]), np.max(r[start_t:(end_t+1),:,1])]
# def init():
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xlim(x_bounds)
#     ax.set_ylim(y_bounds)
#     # sns.despine(top=True, right=True, left=True, bottom=True)
#     # fig.tight_layout()
#     return qv,

# def update(frame):

#     r_t, v_t, theta = frame[:,:2], frame[:,2:4], frame[:,4]
#     qv.set_offsets(r_t)
#     qv.set_UVC(v_t[:,0], v_t[:,1], theta)
#     return qv,

# t_indices = np.arange(start_t+1, end_t+1, step=skip, dtype = int)
# frames = np.concatenate( [r[t_indices], v[t_indices], theta[t_indices][:,:,np.newaxis]], -1)

# anim = FuncAnimation(fig, update, frames = frames,init_func=init,interval=1, blit=True)
# anim.save('test.gif', writer='imagemagick', fps=20)


# r = np.array(r) # try casting to numpy, see if it goes faster
# n_timesteps = r.shape[0]

# start_t, end_t, skip = n_timesteps - 50, n_timesteps-1, 1
# t_indices = np.arange(start_t, end_t, step=skip, dtype = int)

# min_size, max_size = 0., 8.
# t_steps_back = 10
# lwidths = np.linspace(min_size,max_size,t_steps_back-1)
# colors = cm.viridis(np.linspace(0.5,0.8,num=(t_steps_back-1)))[::-1]

# fig, ax= plt.subplots(figsize=(12,20))


# def animate(t):
#     points_all = np.expand_dims(r[segment_indices[t]], 1)
#     segments_all = np.concatenate([points_all[:-1], points_all[1:]], axis = 1) # stack the beginning (t) and end (t+1) of each line-segment, across all agents
#     segments_all_list = np.split(segments_all,N, axis=2) # one array of size (t_steps_back-1, 1, 2) per particle, in a list of such arrays
#     lc = LineCollection(np.vstack(segments_all_list).squeeze(), colors = colors, linewidths=lwidths)
#     ax.add_collection(lc)
#     ax.autoscale_view()

# anim = FuncAnimation(fig, animate, frames = t_indices, interval = 1, blit = False)
# anim.save('test.gif', writer='imagemagick', fps=20)

# def animate(i):

#     axes[0].cla() # clear the previous image
#     axes[1].cla() # clear the previous image

#     im = axes[0].contourf(b_domain, mu_domain, F_landscape, levels=100, cmap=colormap2use)

#     cb = fig.colorbar(im, cax=cax, orientation='vertical', ticks = np.round(np.arange(0,400,step=25)))
#     cb.ax.tick_params(labelsize=22)

#     start_idx = max(0, i - past_tstep)

#     b = x[start_idx:i, b_dim].squeeze()
#     mu = x[start_idx:i, mu_dim].squeeze()
    
#     points = jnp.array([b, mu]).T.reshape(-1,1,2)
#     segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=lwidths,colors = colors)

#     axes[0].add_collection(lc)
#     axes[0].set_xlabel('$b$',fontsize=50,labelpad = 25)
#     axes[0].tick_params(axis='x', labelsize=22)
#     axes[0].set_ylabel('$\mu$',fontsize=50,labelpad = 25)
#     axes[0].tick_params(axis='y', labelsize=22)
#     axes[0].set_title('Free energy $F(b, \mu)$',fontsize=50, pad = 35)


#     axes[1].plot(t_axis[:i],F_over_time[:i], color='r',lw=2.0, label='Free energy: $F(b_t, \mu_t)$')

#     axes[1].set_xlim([0.0, 150])
#     axes[1].set_ylim([0, max_axes1_yvalue])
#     axes[1].set_xlabel('$t$',fontsize=50,labelpad = 25)
#     axes[1].tick_params(axis='x', labelsize=22)
#     axes[1].set_ylabel('$F(b, \mu)$',fontsize=50,labelpad = 25)
#     axes[1].tick_params(axis='y', labelsize=22)
#     axes[1].legend(fontsize=30,loc='upper right')


