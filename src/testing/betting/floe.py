import jax
import jax.numpy as jnp
__all__ = ["floe",]


def floe(payoffs: jax.Array,
         bet_min: float = -0.99,
         bet_max: float = 0.99,
         bisections: int = 20,) -> jax.Array:
    r"""Computes bets based on the Follow the Leave-one-Out E-power strategy.
    Args:
        payoffs: (n,n) leave-one-out payoffs; the rows correspond to time t, and the columns correspond to the leave-out sample l.
        bet_min: minimum allowable bet (default=-0.99).
        bet_max: maximum allowable bet (default=0.99).
        bisections: the number of bisections used for root finding (default=20).
    Returns:
        (n,) predictable sequence of bets.
    """
    assert payoffs.ndim == 2 and payoffs.shape[0] == payoffs.shape[1]
    assert bet_min > -1 and bet_max < 1
    n = payoffs.shape[0]
    uptot = jnp.tri(n, dtype=jnp.bool)
    v = uptot * payoffs  # only consider l <= t

    def bisect(carry, xs):
        lo, hi = carry
        bet = (lo + hi) / 2
        grads = jnp.sum(- v / (1 + bet[:,None] * v), axis=1)
        lo = jnp.where(grads < 0, bet, lo)
        hi = jnp.where(grads < 0, hi, bet)
        return (lo, hi), None

    ones = jnp.ones(n)
    (_, bets), _ = jax.lax.scan(bisect, (bet_min*ones, bet_max*ones), length=bisections)
    return jnp.concat([jnp.zeros(1), bets[:-1]])



