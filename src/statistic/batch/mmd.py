import jax
import jax.numpy as jnp
from utils.type import KernelFn
__all__ = ["mmd2_v",
           "mmd2_v_from_matrices",
           "mmd2_u",
           "mmd2_u_from_matrices",
           "var_h1",
           "var_h1_from_matrices",]


def mmd2_v(X: jax.Array,
           Y: jax.Array,
           kernel: KernelFn,) -> jax.Array:
    assert X.shape == Y.shape
    assert X.ndim == 2
    Kxx = kernel(X, X)
    Kxy = kernel(X, Y)
    Kyy = kernel(Y, Y)
    return mmd2_v_from_matrices(Kxx, Kxy, Kyy)

def mmd2_u(X: jax.Array,
           Y: jax.Array,
           kernel: KernelFn,) -> jax.Array:
    assert X.shape == Y.shape
    assert X.ndim == 2
    Kxx = kernel(X, X)
    Kxy = kernel(X, Y)
    Kyy = kernel(Y, Y)
    return mmd2_u_from_matrices(Kxx, Kxy, Kyy)

def mmd2_v_from_matrices(Kxx: jax.Array,
                         Kxy: jax.Array,
                         Kyy: jax.Array,) -> jax.Array:
    assert Kxx.shape == Kxy.shape == Kyy.shape
    assert Kxx.ndim == 2
    H = Kxx - Kxy - Kxy.mT + Kyy
    return jnp.mean(H)

def mmd2_u_from_matrices(Kxx: jax.Array,
                         Kxy: jax.Array,
                         Kyy: jax.Array,) -> jax.Array:
    assert Kxx.shape == Kxy.shape == Kyy.shape
    assert Kxx.ndim == 2
    n = Kxx.shape[0]
    diag = jnp.eye(n, dtype=jnp.bool)
    H = Kxx - Kxy - Kxy.mT + Kyy
    return jnp.mean(H, where=~diag)



# ===============================
#        Other Statistics
# ===============================

def var_h1_from_matrices(Kxx: jax.Array,
                         Kxy: jax.Array,
                         Kyy: jax.Array,) -> jax.Array:
    assert Kxx.shape == Kxy.shape == Kyy.shape
    assert Kxx.ndim == 2
    H = Kxx - Kxy - Kxy.mT + Kyy
    xi = jnp.mean(H, axis=0)
    mu = jnp.mean(xi)
    return jnp.mean((xi - mu)**2)



