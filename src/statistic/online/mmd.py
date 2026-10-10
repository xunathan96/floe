import jax
import jax.numpy as jnp
from utils.type import KernelFn
__all__ = [# online mmd2
           "online_MMD2",
           "online_MMD2_from_matrices",
           "loo_online_MMD2",
           "loo_online_MMD2_from_matrices",
           # martingale mmd
           "mMMD",
           "mMMD_from_matrices",
           "loo_mMMD",
           "loo_mMMD_from_matrices",
           # online gradient ascent
           "oga",
           "oga_from_matrices",
           # variance of h1
           "online_var_h1",
           "online_var_h1_from_matrices",]

# ===============================
#          Online MMD
# ===============================

def online_MMD2(X: jax.Array,
                Y: jax.Array,
                kernel: KernelFn,) -> jax.Array:
    assert X.shape == Y.shape
    assert X.ndim == 2
    Kxx = kernel(X, X)
    Kxy = kernel(X, Y)
    Kyy = kernel(Y, Y)
    return online_MMD2_from_matrices(Kxx, Kxy, Kyy)

def online_MMD2_from_matrices(Kxx: jax.Array,
                              Kxy: jax.Array,
                              Kyy: jax.Array,) -> jax.Array:
    r"""Computes the V-statistic MMD2 estimator in an online fashion.
    Args:
        Kxx,Kxy,Kyy: (n,n) kernel matrices.
    Returns:
        (n,) array of biased MMD2 loo estimates at each time t=1..n based on up-to-time-t samples
    """
    assert Kxx.shape == Kxy.shape == Kyy.shape
    assert Kxx.ndim == 2
    n = Kxx.shape[0]
    t = jnp.arange(n) + 1
    uptot = jnp.tri(n, dtype=jnp.bool)
    H = Kxx - Kxy - Kxy.mT + Kyy
    sH_tj = jnp.cumsum(H, axis=0)
    ssH_t = jnp.sum(uptot * sH_tj, axis=1)
    return ssH_t / t**2

def loo_online_MMD2(X: jax.Array,
                    Y: jax.Array,
                    kernel: KernelFn,) -> jax.Array:
    assert X.shape == Y.shape
    assert X.ndim == 2
    Kxx = kernel(X, X)
    Kxy = kernel(X, Y)
    Kyy = kernel(Y, Y)
    return loo_online_MMD2_from_matrices(Kxx, Kxy, Kyy)

def loo_online_MMD2_from_matrices(Kxx: jax.Array,
                                  Kxy: jax.Array,
                                  Kyy: jax.Array,) -> jax.Array:
    r"""Computes the V-statistic MMD2 estimator in an online fashion, based on samples 1..t excluding sample l.
    Args:
        Kxx,Kxy,Kyy: (n,n) kernel matrices.
    Returns:
        (n,n) array of biased MMD2 loo estimates; the rows index time from t=1 to t=n, and columns index the leave-out sample l.
    """
    assert Kxx.shape == Kxy.shape == Kyy.shape
    assert Kxx.ndim == 2
    n = Kxx.shape[0]
    uptot = jnp.tri(n, dtype=jnp.bool)
    H = Kxx - Kxy - Kxy.mT + Kyy
    sH_tj = jnp.cumsum(H, axis=0)
    ssH_t = jnp.sum(uptot * sH_tj, axis=1)
    ssH_tl_atleast1_l = 2*sH_tj - jnp.diagonal(H)
    ssH_tl = ssH_t[:,None] - uptot * ssH_tl_atleast1_l
    t = jnp.arange(n) + 1
    Nt = uptot*jnp.clip(t-1, min=1) + ~uptot*t
    return jnp.clip(ssH_tl / Nt**2, min=0)


# ===============================
#         Martingale MMD
# ===============================

def mMMD(X: jax.Array,
         Y: jax.Array,
         kernel: KernelFn,
         radius: float = jnp.sqrt(0.5),
         eps: float = 1e-8,) -> jax.Array:
    assert X.shape == Y.shape
    assert X.ndim == 2
    Kxx = kernel(X, X)
    Kxy = kernel(X, Y)
    Kyy = kernel(Y, Y)
    return mMMD_from_matrices(Kxx, Kxy, Kyy, radius, eps)

def mMMD_from_matrices(Kxx: jax.Array,
                       Kxy: jax.Array,
                       Kyy: jax.Array,
                       radius: float = jnp.sqrt(0.5),
                       eps: float = 1e-8,) -> jax.Array:
    ...


def loo_mMMD(X: jax.Array,
             Y: jax.Array,
             kernel: KernelFn,
             radius: float = jnp.sqrt(0.5),
             eps: float = 1e-8,) -> jax.Array:
    assert X.shape == Y.shape
    assert X.ndim == 2
    Kxx = kernel(X, X)
    Kxy = kernel(X, Y)
    Kyy = kernel(Y, Y)
    return loo_mMMD_from_matrices(Kxx, Kxy, Kyy, radius, eps)

def loo_mMMD_from_matrices(Kxx: jax.Array,
                           Kxy: jax.Array,
                           Kyy: jax.Array,
                           radius: float = jnp.sqrt(0.5),
                           eps: float = 1e-8,) -> jax.Array:
    r"""Computes the martingale MMD statistic by learning the witness RKHS critic on up-to-current samples (<=t), excluding the sample (=l) used for evaluation.
    Args:
        Kxx,Kxy,Kyy: (n,n) kernel matrices.
        radius: Norm of RKHS critic (default=√0.5)
        eps: constant used for numerical stabilization (default=1e-8). 
    Returns:
        (n,n) array of mMMD loo estimates; the rows index time from t=1 to t=n, and columns index the leave-out sample l.
    """
    assert Kxx.shape == Kxy.shape == Kyy.shape
    assert Kxx.ndim == 2
    n = Kxx.shape[0]
    uptot = jnp.tri(n, dtype=jnp.bool)
    H = Kxx - Kxy - Kxy.mT + Kyy
    sH_tj = jnp.cumsum(H, axis=0)
    sH_tl = sH_tj - uptot * jnp.diagonal(H)
    norm2 = loo_online_MMD2_from_matrices(Kxx, Kxy, Kyy)
    norm = jnp.sqrt(jnp.clip(norm2, min=eps))
    t = jnp.arange(n) + 1
    Nt = uptot*jnp.clip(t-1, min=1) + ~uptot*t
    return radius * (sH_tl / Nt) / norm


# ===============================
#     Online Gradient Ascent
# ===============================

def oga(X: jax.Array,
        Y: jax.Array,
        kernel: KernelFn,
        radius: float = jnp.sqrt(0.5),
        lr: float = None,
        eps: float = 1e-8,) -> jax.Array:
    assert X.shape == Y.shape
    assert X.ndim == 2
    Kxx = kernel(X, X)
    Kxy = kernel(X, Y)
    Kyy = kernel(Y, Y)
    return oga_from_matrices(Kxx, Kxy, Kyy, radius, lr, eps)

def oga_from_matrices(Kxx: jax.Array,
                      Kxy: jax.Array,
                      Kyy: jax.Array,
                      radius: float = jnp.sqrt(0.5),
                      lr: float | jax.Array = None,
                      eps: float = 1e-8,) -> jax.Array:
    r"""Computes online gradient ascent on the RKHS critic f to maximize the IPM.
    Args:
        Kxx,Kxy,Kyy: (n,n) kernel matricies.
        radius: radius of the RKHS ball (default=√0.5).
        lr: learning rate used for each step t=1..n; if None we use an adaptive step size (default=None).
        eps: constant used for numerical stabilization (default=1e-8). 
    Returns:
        (n,n) array of differences ft(xj)-ft(yj) where the rows index time t and columns index sample j.
    """
    assert Kxx.shape == Kxy.shape == Kyy.shape
    assert Kxx.ndim == 2
    n = Kxx.shape[0]
    t = jnp.arange(n) + 1
    H = Kxx - Kxy - Kxy.mT + Kyy
    eta = 2*radius / jnp.sqrt(jnp.clip(jnp.cumsum(jnp.diagonal(H)), min=eps))
    if lr is not None:
        eta = jnp.broadcast_to(lr, shape=n)

    def step(carry, xs):
        df, norm = carry    # (n,), scalar
        t, eta, Ht = xs     # scalar, scalar, (n,)
        df_new = df + eta*Ht
        norm_new = jnp.sqrt(jnp.clip(norm**2 + 2*eta*df[t-1] + eta**2*Ht[t-1], min=eps))
        df_new = jnp.where(norm_new > radius, radius*df_new/norm_new, df_new)
        norm_new = jnp.clip(norm_new, max=radius)
        return (df_new, norm_new), df

    dif0 = jnp.zeros(n)
    norm0 = 0.
    _, dfs = jax.lax.scan(step, (dif0, norm0), (t, eta, H))
    return dfs


# ===============================
#         Online Var[h1]
# ===============================

def online_var_h1(X: jax.Array,
                  Y: jax.Array,
                  kernel: KernelFn,) -> jax.Array:
    assert X.shape == Y.shape
    assert X.ndim == 2
    Kxx = kernel(X, X)
    Kxy = kernel(X, Y)
    Kyy = kernel(Y, Y)
    return online_var_h1_from_matrices(Kxx, Kxy, Kyy)

def online_var_h1_from_matrices(Kxx: jax.Array,
                                Kxy: jax.Array,
                                Kyy: jax.Array,) -> jax.Array:
    r"""Estimates (as a V-statistic) the variance of the first order Hoeffding projection in an online fashion.
    Args:
        Kxx,Kxy,Kyy: (n,n) kernel matrices.
    Returns:
        (n,) array of biased variance estimates at times 1..n.
    """
    assert Kxx.shape == Kxy.shape == Kyy.shape
    assert Kxx.ndim == 2
    n = Kxx.shape[0]
    t = jnp.arange(n) + 1
    uptot = jnp.tri(n, dtype=jnp.bool)
    H = Kxx - Kxy - Kxy.mT + Kyy
    sH_tj = jnp.cumsum(H, axis=0)
    ssH_t = jnp.sum(uptot * sH_tj, axis=1)
    mH_tj = sH_tj / t[:,None]
    mmH_t = ssH_t / t**2
    return jnp.sum(uptot * (mH_tj - mmH_t[:,None])**2, axis=1) / t
    # return jnp.mean((mH_tj - mmH_t[:,None])**2, axis=1)  # this is incorrect!





