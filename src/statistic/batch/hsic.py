import jax
import jax.numpy as jnp
from utils.type import KernelFn
__all__ = ["hsic_v",
           "hsic_v_from_matrices",]


def hsic_v(X: jax.Array,
           Y: jax.Array,
           kernel_x: KernelFn,
           kernel_y: KernelFn) -> jax.Array:
    assert X.shape == Y.shape and X.ndim == 2
    K = kernel_x(X,X)
    L = kernel_y(Y,Y)
    return hsic_v_from_matrices(K, L)

def hsic_v_from_matrices(K: jax.Array,
                         L: jax.Array) -> jax.Array:
    r"""Computes the V-statistic estimate of the Hilbert-Schmidt Independence Criterion between paired samples X and Y
    Args:
        K: (n,n) kernel matrix for X
        L: (n,n) kernel matrix for Y
    Returns:
        res: scalar HSIC value
    """
    assert K.ndim == L.ndim == 2
    assert K.shape == L.shape and K.shape[0] == K.shape[1]
    n = K.shape[0]
    S1 = jnp.sum(K*L)
    S2 = jnp.sum(K.sum(0) * L.sum(0))
    S3 = jnp.sum(K) * jnp.sum(L)
    # work in log-space for numerical stability
    M1 = jnp.sign(S1) * jnp.exp(jnp.log(jnp.abs(S1)) - 2*jnp.log(n))
    M2 = jnp.sign(S2) * jnp.exp(jnp.log(jnp.abs(S2)) - 3*jnp.log(n))
    M3 = jnp.sign(S3) * jnp.exp(jnp.log(jnp.abs(S3)) - 4*jnp.log(n))
    return jnp.clip(M1 - 2*M2 + M3, min=0)


