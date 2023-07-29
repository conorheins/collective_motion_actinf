from jax import numpy as jnp
from jax import random, lax, vmap
from jax.scipy.linalg import block_diag
import numpy as np
from genmodel import create_temporal_precisions, create_temporal_precisions_numpy
from scipy.special import factorial, gamma

def create_dt_matrix(dt, num_taylor_pns = 3, num_do = 3):
    """
    A matrix of expansion coefficients when performing Eulerian integration of a differential equation
    """
    # generate Taylor polynomial update matrix
    pn_coeffs = dt**(np.arange(1,num_taylor_pns+1)) * (1.0 / factorial(np.arange(1,num_taylor_pns+1))) # create vector of coefficients for use in Taylor polynomial expansion
    dt_matrix = np.zeros((num_do,num_taylor_pns)) # preallocate the matrix

    for o_i in range(num_do):
        dt_matrix[o_i,o_i:num_taylor_pns] = pn_coeffs[:(num_taylor_pns-o_i)] # create shift matrix of taylor polynomials - this means that the generalised integration equation can be simplified to a matrix multiply

    if dt_matrix.shape[1] > dt_matrix.shape[0]:
        dt_matrix = dt_matrix[:,:num_do] # truncate the number of polynomial terms to the number of dynamical orders being used (can't use a jerk term in the taylor expansion if it doesn't exist as a belief)

    return jnp.array(dt_matrix)

def sample_noise(alpha, noise_magnitude, desired_smoothness, noise_type, state_dim, orders_of_motion, dt, key, n_samples):
    """
    Samples a random trajectory from a generalised OU process with desired smoothness or autocorrelation given by `desired_smoothness`
    """

    vec_dim = state_dim * orders_of_motion

    dt_matrix = create_dt_matrix(dt, num_taylor_pns = orders_of_motion, num_do = orders_of_motion)
    dt_matrix = jnp.kron(dt_matrix,jnp.eye(state_dim))

    Sigma_time = jnp.array(create_temporal_precisions_numpy(truncation_order = orders_of_motion, smoothness = desired_smoothness, form = noise_type)[1])

    # Sigma_time  = jnp.array(create_temporal_precisions(truncation_order = orders_of_motion, smoothness = desired_smoothness, form = noise_type)[1])
    Sigma_total = jnp.kron(Sigma_time, noise_magnitude * jnp.eye(state_dim))

    A0 = alpha * jnp.eye(state_dim)

    # create the generalised flow by creating block diagonal matrix of the flow at each order. Flow at the highest order is left empty because the 'descending' expectation D\tilde{\boldsymbol{\mu}} will also be 0, so 'descending' state prediction errors at highest order
    # A_blocks = [A0 for _ in range(orders_of_motion-1)] + [jnp.zeros((state_dim,state_dim))]  # implies local linearization at higher orders
    A_blocks = [A0 for _ in range(orders_of_motion)] # implies local linearization at higher orders
    tilde_A = block_diag(*A_blocks)

    def f_deterministic(x):
        # generalized flow function
        return -tilde_A @ x

    noise_samples = random.multivariate_normal(key, shape = (n_samples,), mean = jnp.zeros(vec_dim), cov = Sigma_total)

    def integration_step(carry, t):
        x_next = carry + dt_matrix @ (f_deterministic(carry) + noise_samples[t])

        return x_next, x_next

    x_init = noise_samples[0]
    # print(x_init[state_dim])

    _, sampled_trajectory = lax.scan(integration_step, x_init, jnp.arange(1, n_samples))

    return sampled_trajectory[:,:state_dim]

def generate_path_from_gen_coord(alpha, noise_magnitude, desired_smoothness, noise_type, state_dim, orders_of_motion, dt, key, x0, n_seconds):
    """
    Samples an analytic trajectory of a "generalised" linear Langevin equation (i.e. an OU process) with desired smoothness or autocorrelation given by `desired_smoothness`
    """

    vec_dim = state_dim * orders_of_motion

    Sigma_time = jnp.array(create_temporal_precisions_numpy(truncation_order = orders_of_motion, smoothness = desired_smoothness, form = noise_type)[1])
    Sigma_total = jnp.kron(Sigma_time, noise_magnitude * jnp.eye(state_dim))

    if state_dim > 1:
        A0 = alpha * jnp.eye(state_dim) + 0.5*alpha*jnp.eye(state_dim, k=1) - 0.5*alpha*jnp.eye(state_dim, k=-1)
    else:
        A0 = alpha * jnp.eye(state_dim)

    A_blocks = [A0 for _ in range(orders_of_motion)] # implies local linearization at higher orders
    tilde_A = block_diag(*A_blocks)

    def f_deterministic(x0):
        # generalized flow function
        return -A0 @ x0

    noise_samples = random.multivariate_normal(key, mean = jnp.zeros(vec_dim), cov = Sigma_total)
    _, next_key = random.split(key)

    t_axis = np.arange(0.0, n_seconds, step=dt)
    x_t = np.zeros((state_dim, len(t_axis)))

    x_t[:,0] = x0

    # generate successive derivatives from the position (i.e. differentiate the position-SDE to get velocity-SDE, etc)
    new_gen_state = np.zeros_like(noise_samples)
    new_gen_state[:state_dim] = x0

    for order_i in range(1, orders_of_motion):
        new_gen_state[(state_dim*order_i):(state_dim*(order_i+1))] = -A_blocks[order_i-1] @ new_gen_state[(state_dim*(order_i-1)):(state_dim*order_i)] + noise_samples[(state_dim*(order_i-1)):(state_dim*order_i)]
    
    last_index = 0
    expansion_duration = 500
    for i in range(1, len(t_axis)):       
        
        t = t_axis[i] - t_axis[last_index]
       
        # if enough timesteps have elapsed, reset the new generalised coordinates
        if i % expansion_duration == 0:
            last_index = i

            new_gen_state[:state_dim] = x_t_position # overwrite the new_gen_state vector with the position

            noise_samples = random.multivariate_normal(next_key, mean = jnp.zeros(vec_dim), cov = Sigma_total)
            next_key, _ = random.split(next_key)

            # generate successive derivatives from the position (i.e. differentiate the position-SDE to get velocity-SDE, etc)
            for order_i in range(1, orders_of_motion):
                new_gen_state[(state_dim*order_i):(state_dim*(order_i+1))] = -A_blocks[order_i-1] @ new_gen_state[(state_dim*(order_i-1)):(state_dim*order_i)] + noise_samples[(state_dim*(order_i-1)):(state_dim*order_i)]

        # this is the Taylor expansion of the position using the higher orders
        x_t_position = np.zeros(state_dim)
        for order_i in range(1, orders_of_motion):
            x_t_position += (new_gen_state[(state_dim*order_i):(state_dim*(order_i+1))] * (t**order_i)) / factorial(order_i)

        x_t[:,i] = x_t_position

    
    # how to integrate a noise path
    # omega_t = np.zeros((state_dim, len(t_axis)))
    # for (i, t) in enumerate(t_axis):
    #     omega_t_position = noise_samples[:state_dim]
    #     for order_i in range(1, orders_of_motion):
    #         omega_t_position += (noise_samples[(state_dim*order_i):(state_dim*(order_i+1))] * (t**order_i)) / factorial(order_i)
    #     omega_t[:,i] = omega_t_position
    # return t_axis, x_t, omega_t
    return t_axis, x_t


def generate_path_from_gen_coord_old(alpha, noise_magnitude, desired_smoothness, noise_type, state_dim, orders_of_motion, dt, key, n_seconds):
    """
    Samples a random trajectory of smooth noise using the "static equation" of noise $(D-J)^{-1}\tilde{\omega}$ and a Taylor integration.
    """

    vec_dim = state_dim * orders_of_motion

    Sigma_time = jnp.array(create_temporal_precisions_numpy(truncation_order = orders_of_motion, smoothness = desired_smoothness, form = noise_type)[1])

    Sigma_total = jnp.kron(Sigma_time, noise_magnitude * jnp.eye(state_dim))

    A0 = alpha * jnp.eye(state_dim)

    # create the generalised flow by creating block diagonal matrix of the flow at each order. Flow at the highest order is left empty because the 'descending' expectation D\tilde{\boldsymbol{\mu}} will also be 0, so 'descending' state prediction errors at highest order
    # A_blocks = [A0 for _ in range(orders_of_motion-1)] + [jnp.zeros((state_dim,state_dim))]  # implies local linearization at higher orders
    A_blocks = [A0 for _ in range(orders_of_motion)] # implies local linearization at higher orders
    tilde_A = block_diag(*A_blocks)

    noise_samples = random.multivariate_normal(key, mean = jnp.zeros(vec_dim), cov = Sigma_total)

    # D_shift = jnp.diag(jnp.ones((state_dim*orders_of_motion- state_dim)), k = state_dim)

    # generalised_state = jnp.linalg.inv(D_shift + tilde_A) @ noise_samples

    # use recursive Taylor polynomials to generate the (n-1)-th order from the n-th order, all the way down to integrate the position

    t_axis = np.arange(0.0, n_seconds, step=dt)
    # path_v1 = np.zeros_like(t_axis)
    path_v2 = np.zeros_like(t_axis)
    # path_v3 = np.zeros_like(t_axis)

    # dt_matrix = create_dt_matrix(dt, num_taylor_pns=orders_of_motion, num_do=orders_of_motion) 
    # print(dt_matrix)

    x0 = 0.0

    # now do it using the way explained in Lance's overleaf, evaluating \tilde{x}_0
    # new_gen_state = np.zeros_like(noise_samples)
    # new_gen_state[0] = x0
    # for order_i in range(1, orders_of_motion):
    #     new_gen_state[order_i] = -alpha * new_gen_state[order_i-1] + noise_samples[order_i-1]
    
    path_v2[0] = x0
    for (i, t) in enumerate(t_axis):

        # # generate series of polynomial coefficients through the relation coeff^n = t**n / n!, where n is the order of differentiation (temporal embedding order)
        # dt_matrix = create_dt_matrix(t, num_taylor_pns=orders_of_motion, num_do=orders_of_motion)

        # # integrate down each x^{n} using integration from the "higher" orders
        # x_int = dt_matrix @ generalised_state # the [:-1] is because we want to fix the highest order while changing the remaining (lower) orders

        # path_v1[i] = x_int[0] # take just the 0-th order of the generalised state (the position)

        # do it explicitly via for-loop as well for validation against the matrix-multiply version

        # x_t = generalised_state[-1] # start at the highest order (e.g. the acceleration, when `orders_of_motion == 3`)
        # new_gen_state = np.zeros_like(generalised_state)
        # new_gen_state[-1] = generalised_state[-1]

        # # this is a loop over each entry of `new_gen_state`, in which each temporal derivative gets updated
        # for order_i in range(orders_of_motion-1, 0, -1): 
        #     # x_t += (x_t * (t**order_i)) / factorial(order_i)
        #     # for each temporal derivative, run another loop that "starts from the top" (the highest temporal order, which is fixed) and integrates down to derivative with index `order_i`
        #     x_order_i = 0.
        #     for order_j in range(orders_of_motion, order_i, -1): # when order_j == 0, this should integrate to the position at time t, using the velocity, acceleration, jerk, etc...
        #         x_order_i += (new_gen_state[order_j] * (t**order_j)) / factorial(order_j)
        #     new_gen_state[order_i] = x_order_i

        # path_v2[i] = new_gen_state[0]

        new_gen_state[0] = path_v2[0]
        for order_i in range(1, orders_of_motion):
            new_gen_state[order_i] = -alpha * new_gen_state[order_i-1] + noise_samples[order_i-1]
        
       
        # this is a loop over each entry of `new_gen_state`, in which each temporal derivative gets updated
        new_position = x0
        for order_i in range(1, orders_of_motion):
            
            new_position += (new_gen_state[order_i] * (t**order_i)) / factorial(order_i)

        path_v2[i] = new_position

        # this is a loop over each entry of `generalised_state`, in which each temporal derivative gets updated
        # new_position = 0.
        # for order_i in range(orders_of_motion-1, -1, -1): 
            
        #     # if i == 1:
        #     #     print(t)
        #     #     coeff = (t**order_i) / factorial(order_i)
        #     #     print(f'Taylor coefficient for order {order_i}: {coeff}\n')
        #     new_position += (generalised_state[order_i] * (t**order_i)) / factorial(order_i)

        # path_v3[i] = new_position
    
    return t_axis, path_v2
    # return t_axis, path_v1, path_v3

        









    

    



