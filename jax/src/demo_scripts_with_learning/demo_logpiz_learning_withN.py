from itertools import product
import jax
from jax import random, lax, vmap
from jax import numpy as jnp
import numpy as np
from funcy import merge
from scipy import stats
import numpy as np

from genprocess import init_gen_process, compute_sector_dists_over_time
from utils import get_default_inits, make_single_timestep_fn, initialize_meta_params
from learning import make_dfdparams_fn, reparameterize
from genmodel import init_genmodel, create_temporal_precisions
from matplotlib import pyplot as plt

key = random.PRNGKey(3)

cpus = jax.devices("cpu")
gpus = jax.devices("gpu")

T, dt, D = 125, 0.01, 2

n_precision_levels = 10

n_trials = 20

gp_smoothness = 1.0
gp_logprecisions = jnp.linspace(-2.0, 2.0, n_precision_levels)
groupsizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

n_groupsize_levels = len(groupsizes)
n_total_levels = n_precision_levels * n_groupsize_levels

burn_in_time = 2000
optimal_area = 3.86438 # this is the area of the bounding box that, once initialized, will lead the average nearest neighbour distance between agents to be 1.0
s_l = jnp.sqrt(optimal_area) # side of square length

parameter_combos = list(product(gp_logprecisions, groupsizes))

key_list = random.split(key, num=n_total_levels)

learned_logpiz_vals = np.zeros((n_precision_levels, n_groupsize_levels, n_trials))
average_nn_dist = np.zeros((n_precision_levels, n_groupsize_levels, n_trials))

for ii, (combo_i, key_i) in enumerate(zip(parameter_combos, key_list)):

    log_precis, N = combo_i

    row_id = int(jnp.where(gp_logprecisions == log_precis)[0][0])
    col_id = int(jnp.where(N == jnp.array(groupsizes))[0][0])

    init_dict = get_default_inits(N, T, dt)
    init_dict['posvel_init']['pos_x_bounds'] = [-s_l*0.5, s_l*0.5] # figure out an expression in terms of N
    init_dict['posvel_init']['pos_y_bounds'] = [-s_l*0.5, s_l*0.5] # figure out an expression in terms of N
    zh_variance = jnp.exp(-log_precis)
    zhprime_variance = zh_variance / (2*gp_smoothness**2)

    init_dict['z_h'] = zh_variance
    init_dict['z_hprime'] = zhprime_variance

    trial_i = 0
    trial_key = key_i

    while trial_i < n_trials:
        _, trial_key = random.split(trial_key, 2)
        pos, vel, genproc, new_key = init_gen_process(trial_key, init_dict)
        steady_state_time = int(0.5 *len(genproc['t_axis']))

        genmodel = init_genmodel(init_dict)

        meta_params = initialize_meta_params(infer_lr = 0.1, 
                                            nsteps_infer = 1, 
                                            action_lr = 0.1, 
                                            nsteps_action = 1, 
                                            learning_lr = 0.001, 
                                            nsteps_learning = 1, 
                                            normalize_v = True
                                            )
        
        ## Parameterization specific to learning f_params
        _, piz_key = random.split(new_key)

        logpiz_spatial_all = 0.5 * random.normal(piz_key, shape = (N,)) # sample a different sensory (log) spatial precision for every agent

        Pi_z_temporal, _ = create_temporal_precisions(truncation_order=genmodel['ndo_phi'], smoothness=init_dict['s_z']) # technically correct, but need to decrease smoothness s_z to make it look ok

        def parameterize_Pi_z(logpi_z_spatial):
            """
            Parameterize Pi_z (sensory preicison matrix) with `s_z` under assumption of only two orders of motion
            """
            Pi_z_spatial = jnp.exp(logpi_z_spatial) * jnp.eye(genmodel['ns_phi'])
            return jnp.kron(Pi_z_temporal, Pi_z_spatial)
        
        preparams = {'logpiz_spatial': logpiz_spatial_all}
        parameterization_mapping = {'logpiz_spatial': {'to_arg_name': 'Pi_z', 'fn': parameterize_Pi_z}}                                    

        initial_learnable_params = vmap(reparameterize, (0, None))(preparams, parameterization_mapping)
        genmodel = merge(genmodel, initial_learnable_params)

        mu = genmodel['f_params']['tilde_eta'].copy().reshape(N, genmodel['ndo_x']*genmodel['ns_x']).T

        dFdparam_function = make_dfdparams_fn(genmodel, preparams, parameterization_mapping, N)
        single_timestep = make_single_timestep_fn(genproc, genmodel, dFdparam_function, parameterization_mapping, meta_params)
        def step_fn(carry, t):
            pos_past, vel_past, mu_past, preparams_past = carry
            pos, vel, mu, preparams, F = single_timestep(pos_past, vel_past, mu_past,preparams_past, t)
            return (pos, vel, mu, preparams), (pos, vel, mu, preparams['logpiz_spatial'], F)
        init_state = (pos, vel, mu, preparams)
        out, history = lax.scan(step_fn, init_state, jnp.arange(len(genproc['t_axis'])))
        position_history, velocity_history, logpiz_history = history[0], history[1], history[3]

        if N > 100:
            position_history = jax.device_put(position_history, cpus[0])
            velocity_history = jax.device_put(velocity_history, cpus[0])

        good_fish_ids = []
        check_time = min(5*burn_in_time, steady_state_time)
        for n_idx in range(N):
            ratio = jnp.absolute(logpiz_history[-1,n_idx] / logpiz_history[burn_in_time,n_idx])
            if ratio < 5.0:
                good_fish_ids.append(n_idx)

        good_fish_ids = jnp.array(good_fish_ids)

        if len(good_fish_ids) < int(0.5 * N):
            print(f'Warning, trial {trial_i} with group size {N} and sensory variance {jnp.exp(-log_precis)} has less than half good ids...\n')
            print(f'Therefore, not logging results and running an extra trial\n')
        else:
            learned_logpiz_vals[row_id, col_id, trial_i] = logpiz_history[steady_state_time:,good_fish_ids].mean(-1).mean() # average learned value of the log_piz parameter, averaged across agents and timesteps
            h_hist = compute_sector_dists_over_time(position_history, velocity_history, genproc)
            average_nn_dist[row_id, col_id, trial_i] = jnp.nanmean(h_hist[steady_state_time:,:,good_fish_ids],axis=1).mean()
            trial_i += 1

np.save("learned_logpiz_vals.npy", learned_logpiz_vals)
np.save("average_nn_dist_logpiz.npy", average_nn_dist)

# fig, axes = plt.subplots(1,2,figsize = (10,8))

# across_trial_mean = np.nanmean(learned_logpiz_vals, axis = -1)
# across_trial_std = np.nanstd(learned_logpiz_vals, axis = -1)
# upper_bound, lower_bound = across_trial_mean + 2.5 * across_trial_std, across_trial_mean - 2.5 * across_trial_std
# axes[0].imshow(across_trial_mean)

# across_trial_mean = np.nanmean(average_nn_dist, axis = -1)
# across_trial_std = np.nanstd(average_nn_dist, axis = -1)
# upper_bound, lower_bound = across_trial_mean + 2.5 * across_trial_std, across_trial_mean - 2.5 * across_trial_std
# axes[1].imshow(across_trial_mean)

# plt.show()
