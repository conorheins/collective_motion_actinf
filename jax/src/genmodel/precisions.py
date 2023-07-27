import jax.numpy as jnp
from jax import lax
import numpy as np

def create_temporal_precisions_numpy(truncation_order = 1, smoothness = 1.0, form = 'Gaussian'):
    """
    Direct numpy re-implementation of the SPM function 'spm_DEM_R.m'
    Returns the precision of the temporal derivatives of a Gaussian process
    FORMAT R,V = create_temporal_precisions(truncation_order,smoothness,form)
    n    - truncation order [default: 1]
    s    - temporal smoothness - s.d. of kernel {bins} [default: 1.0]
    form - 'Gaussian', '1/f' [default: 'Gaussian']
    R    - shape (n,n)     E*V*E: precision of n derivatives
    V    - shape (n,n)     V:    covariance of n derivatives
    """
    if form == 'Gaussian':
        k = np.arange(0,truncation_order)
        r = np.zeros(1+2*k[-1])
        x = np.sqrt(2.0) * smoothness
        r[2*k] = np.cumprod(1 - (2*k))/(x**(2*k))
    elif form == '1/f':
        k = np.arange(0,truncation_order)
        x = 8.0*smoothness**2
        r[2*k] = (-1)**k * gamma(2*k + 1)/(x**(2*k))

    V = np.zeros((truncation_order, truncation_order))
    for i in range(truncation_order):
        V[i,:] = r[np.arange(0,truncation_order) + i]
        r = -r

    R = np.linalg.inv(V)

    return R, V

# def create_temporal_precisions_v2(truncation_order = 1, smoothness = 1.0):
#     """
#     JAX functional re-implementation of `spm_DEM_R.m`
#     Returns the precision of the temporal derivatives of a Gaussian process
#     FORMAT R,V = create_temporal_precisions(truncation_order,smoothness,form)
#     n    - truncation order [default: 1]
#     s    - temporal smoothness - s.d. of kernel {bins} [default: 1.0]
#     form - 'Gaussian', '1/f' [default: 'Gaussian']
#     R    - shape (n,n)     E*V*E: precision of n derivatives
#     V    - shape (n,n)     V:    covariance of n derivatives
#     """

#     k = jnp.arange(0, truncation_order)
#     r = jnp.zeros(1 + 2*k[-1])
#     x = jnp.sqrt(2.0) * smoothness
#     cumprod = jnp.cumprod(1 - (2*k))/(x**(2*k))
#     r = r.at[2*k].set(jnp.cumprod(1 - (2*k))/(x**(2*k))) # this is fine actually for passing gradients

#     V = jnp.zeros((truncation_order, truncation_order)) # so is this, apparently
#     for i in range(truncation_order):
#         V = V.at[i,:].set(r[jnp.arange(0,truncation_order) + i])
#         r = -r

#     R = jnp.linalg.inv(V)

#     return R, V


def create_temporal_precisions(truncation_order = 1, smoothness = 1.0):
    """
    JAX functional re-implementation of `spm_DEM_R.m`
    Returns the precision of the temporal derivatives of a Gaussian process
    FORMAT R,V = create_temporal_precisions(truncation_order,smoothness,form)
    n    - truncation order [default: 1]
    s    - temporal smoothness - s.d. of kernel {bins} [default: 1.0]
    form - 'Gaussian', '1/f' [default: 'Gaussian']
    R    - shape (n,n)     E*V*E: precision of n derivatives
    V    - shape (n,n)     V:    covariance of n derivatives
    """

    k = jnp.arange(0, truncation_order)

    # def build_r(carry, i):
    #     blah = carry
    #     return blah, 0.0
    # _, r = lax.scan(build_r, jnp.array([0.0]), jnp.arange(1 + 2*k[-1]))
    
    r = jnp.zeros(1 + 2*k[-1])
    x = jnp.sqrt(2.0) * smoothness
    cumprod = jnp.cumprod(1 - (2*k))/(x**(2*k))
    # r = r.at[2*k].set(cumprod) # this is fine for passing gradients actually

    r = jnp.vstack((r[::2],cumprod)).reshape((-1,),order='F')[1:]

    def build_v(carry, i):
        r = carry
        v_curr = r[jnp.arange(0, truncation_order) + i]
        r = -r
        return r, v_curr

    _, v_rows = lax.scan(build_v, r, jnp.arange(truncation_order))

    V = jnp.stack(v_rows)
    R = jnp.linalg.inv(V)

    return R, V

def create_full_precision_matrix(num_states, num_do, spatial_precision = 1.0, smoothness = 1.0):
    """ 
    Factorized parameterization of a precision matrix using the outer kronecker product of 
    a temporal precision matrix (inverse of the covariance matrix between generalised fluctuations at different orders)
    and a spatial precision matrix (inverse of the covariance matrix between fluctuations at different spatial dimensions
    """
    
    Pi_spatial = spatial_precision * jnp.eye(num_states)
    Pi_temporal = create_temporal_precisions(num_do, smoothness)[0]
    Pi_full = jnp.kron(Pi_temporal, Pi_spatial)

    return Pi_full


