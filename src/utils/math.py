import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable


@eqx.filter_jit
def bisection_method(fun: Callable[[float | jax.Array], float | jax.Array],
                     amin: float | jax.Array,
                     amax: float | jax.Array,
                     monotonicity: str = 'increasing',
                     num_bisections: int = 100,):
    r"""compute the root of the given function in the interval [amin,amax] via the bisection method.
    Args:
        fun: function f that we want to find the root of; its output must be the same shape as its input.
        amin: the lower bound of the domain of f with the same shape as the function input/output.
        amax: the upper bound of the domain of f with the same shape as the function input/output.
        monotonicity: either 'increasing' or 'decreasing' (default='increasing').
        num_bisections: the number of bisections used in root finding (default=100).
    Returns:
        the estimated root with same shape as function input/output.
    """
    def bisect(carry, _):
        umin, umax = carry
        x = (umin + umax) / 2.
        y = fun(x)
        if monotonicity == 'increasing':
            umin = jnp.where(y < 0., x, umin)
            umax = jnp.where(y < 0., umax, x)
        elif monotonicity == 'decreasing':
            umin = jnp.where(y > 0., x, umin)
            umax = jnp.where(y > 0., umax, x)
        return (umin, umax), None
    (umin, umax), _ = jax.lax.scan(bisect, (amin, amax), length=num_bisections)
    return umax


@eqx.filter_jit
def level_to_quantile(x: jax.Array,
                      alpha: float | jax.Array,
                      xmin: float = None,
                      xmax: float = None,
                      mode: str = 'lower',):
    r"""computes the empirical quantile estimator based on order statistics.
    Args:
        x: (n,) array of samples
        alpha: (b,) array of quantile levels to estimate
        xmin: minimum value that samples x can take; if None then use x.min()
        xmin: maximum value that samples x can take; if None then use x.max()
        mode: use either the `lower` or `upper` empirical quantile
    Returns:
        the estimated empirical quantile at the given alpha levels.
    """
    alpha = jnp.atleast_1d(alpha)
    assert x.ndim == alpha.ndim == 1
    n = x.shape[0]
    xsort = jnp.sort(x)
    if xmin is None: xmin = xsort[0]
    if xmax is None: xmax = xsort[-1]
    if mode == 'lower':
        rank = jnp.ceil(alpha*n).astype(jnp.int32)
        qa = xsort[rank-1]
    elif mode == 'upper':
        rank = jnp.floor(alpha*n).astype(jnp.int32)
        qa = jnp.where(rank <= 0, xsort[0], xsort[rank-1])
    qa = jnp.where(alpha <= 0., xmin, qa)
    qa = jnp.where(alpha >= 1., xmax, qa)
    return qa



