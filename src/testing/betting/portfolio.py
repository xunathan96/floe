import jax
import jax.numpy as jnp
__all__ = ["universal_portfolio",]


def universal_portfolio(payoffs: jax.Array,
                        bet_min: float = -0.99,
                        bet_max: float = 0.99,
                        prior: str = "uniform",
                        n_portfolios: int = 100,):
    assert payoffs.ndim == 1
    assert -1 < bet_min <= 0 and 0 <= bet_max < 1
    assert prior in ("uniform", "beta")
    n = payoffs.shape[0]

    xrange = jnp.arange(n_portfolios+1) / n_portfolios
    b = bet_min + (bet_max - bet_min) * xrange   # (p,)

    if prior == "uniform":
        pb = jnp.broadcast_to(1 / (bet_max - bet_min), shape=b.shape)
    elif prior == "beta":
        # beta(1/2,1/2) prior on each bet < 0 and bet > 0 half
        pos = jnp.pi * jnp.sqrt(b * (bet_max - b))
        neg = jnp.pi * jnp.sqrt(-b * (-bet_min + b))
        compose = jnp.where(b >= 0, pos, neg)
        pb = 1 / jnp.clip(compose, min=0.01)
        if bet_min < 0: pb /= 2  # reweight the mixture

    logW = jnp.log(1 + b * payoffs[:,None]).cumsum(0)    # (n,p)
    Wb = jnp.exp(logW) * pb
    bets = jnp.sum(b * Wb, axis=1) / jnp.sum(Wb, axis=1)
    return jnp.concat([jnp.zeros(1), bets[:-1]])



