import jax
import jax.numpy as jnp
import equinox as eqx
__all__ = ['gaussian',
           'gaussian_1d',
           'pairwise_distance',
           'median_heuristic',
           'online_median_heuristic',]


# ====================================
#             Functions
# ====================================

@eqx.filter_jit
def gaussian(x: jax.Array,
             y: jax.Array,
             bandwidth: jax.Array,
             scale: jax.Array = 1.,):
    assert (x.ndim == y.ndim == 1) \
        and (isinstance(bandwidth, float) or bandwidth.ndim == 0) \
        and (isinstance(scale, float) or scale.ndim == 0)
    return scale * jnp.exp(- 0.5 * jnp.sum((x - y)**2) / bandwidth**2)


@eqx.filter_jit
def gaussian_1d(x: jax.Array,
                y: jax.Array,
                bandwidth: jax.Array,
                scale: jax.Array,):
    assert x.ndim == y.ndim == bandwidth.ndim == scale.ndim == 0
    return scale * jnp.exp(-0.5 * ((x-y)/bandwidth)**2 )


@eqx.filter_jit
def pairwise_distance(X: jax.Array, Y: jax.Array):
    """Computes the euclidean pairwise distances between samples in X and Y
    Args:
        X: (m,d) array
        Y: (n,d) array
    Returns:
        the (m,n) array of euclidean distances |xi - yj|
    """
    assert (X.ndim == Y.ndim == 2) and (X.shape[1] == Y.shape[1])
    x2 = jnp.sum(X**2, axis=1, keepdims=True)   # (m,1)
    y2 = jnp.sum(Y**2, axis=1)                  # (n)
    xy = X @ Y.mT                               # (m,n)
    dist2 = x2 - 2*xy + y2                      # (m,n)
    return jnp.sqrt(jnp.clip(dist2, min=0.))


def median_heuristic(X: jax.Array, Y: jax.Array = None):
    Y = X if Y is None else Y
    use_self = True if Y is None else False
    assert X.ndim == Y.ndim == 2
    m = X.shape[0]
    n = Y.shape[0]
    Dxy = pairwise_distance(X, Y)   # (m,n)
    if use_self:
        mask = jnp.tri(m, n, k=-1)
        Dxy_unique = jnp.where(mask, Dxy, jnp.nan)
        median = jnp.nanmedian(Dxy_unique)
    else:
        median = jnp.median(Dxy)
    return median / jnp.sqrt(2)


def online_median_heuristic(X: jax.Array, Y: jax.Array = None):
    Y = X if Y is None else Y
    use_self = True if Y is None else False
    assert X.shape == Y.shape
    n = X.shape[0]
    Dxy = pairwise_distance(X, Y)   # (n,n)
    def median_at_t(carry, t):
        i = jnp.arange(n)
        uptot = (i <= t)[:,None] * (i <= t)  # (n,n)
        if use_self:
            mask = jnp.tri(n, n, k=-1)
            Dxy_hist = jnp.where(uptot*mask, Dxy, jnp.nan)
        else:
            Dxy_hist = jnp.where(uptot, Dxy, jnp.nan)
        median = jnp.nanmedian(Dxy_hist)
        return None, median
    _, medians = jax.lax.scan(median_at_t, None, jnp.arange(n)) # (n,)
    return medians / jnp.sqrt(2)    # (n,)


