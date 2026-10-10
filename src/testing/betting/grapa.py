import jax
import jax.numpy as jnp
__all__ = ["grapa",]


def grapa(payoffs: jax.Array,
          bet_min: float = -0.99,
          bet_max: float = 0.99,
          bisections: int = 20,) -> jax.Array:
    assert payoffs.ndim == 1
    assert -1 < bet_min <= 0 and 0 <= bet_max < 1
    n = payoffs.shape[0]
    uptot = jnp.tri(n, dtype=jnp.bool)

    def bisect(carry, xs):
        lo, hi = carry  # (n,), (n,)
        bet = (lo + hi) / 2
        g = - payoffs / (1 + bet[:,None] * payoffs)
        grads = jnp.sum(uptot*g, axis=1)
        lo = jnp.where(grads < 0, bet, lo)
        hi = jnp.where(grads < 0, hi, bet)
        return (lo, hi), None

    ones = jnp.ones(n)
    (_, bets), _ = jax.lax.scan(bisect, (bet_min*ones, bet_max*ones), length=bisections)
    return jnp.concat([jnp.zeros(1), bets[:-1]])






