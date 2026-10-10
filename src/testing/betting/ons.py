import jax
import jax.numpy as jnp
__all__ = ["ons",
           "ons_with_reg",]


def ons(payoffs: jax.Array,
        gamma: float = None,
        a0: float = 1.,
        bet_min: float = 0.,
        bet_max: float = 0.99,) -> jax.Array:
    assert payoffs.ndim == 1
    assert -1 < bet_min <= 0 and 0 <= bet_max < 1
    # default gamma bound (assume payoffs are positive: true in the limit)
    b = (bet_max - bet_min) / (1 + bet_min)
    gamma = gamma or 2*(b - jnp.log(1+b)) / (b**2)

    def step(carry, v):
        bet, a = carry
        g = - v / (1 + bet*v)
        a += g**2
        bet = jnp.clip(bet - g / (gamma*a), min=bet_min, max=bet_max)
        return (bet, a), bet

    _, bets = jax.lax.scan(step, (0., a0), payoffs)
    return jnp.concat([jnp.array([0.]), bets[:-1]], axis=0)


def ons_with_reg(payoffs: jax.Array,
                 gamma: float = None,
                 a0: float = 1.,
                 bet_min: float = 0.,
                 bet_max: float = 0.99,
                 reg_floors: jax.Array = None,
                 reg_ceils: jax.Array = None,) -> jax.Array:
    assert payoffs.ndim == 1
    assert -1 < bet_min <= 0 and 0 <= bet_max < 1
    assert reg_floors is None or (reg_floors.shape == payoffs.shape)
    assert reg_ceils is None or (reg_ceils.shape == payoffs.shape)
    n = payoffs.shape[0]
    bmins = jnp.broadcast_to(bet_min, shape=n) if reg_floors is None else reg_floors
    bmaxs = jnp.broadcast_to(bet_max, shape=n) if reg_ceils is None else reg_ceils

    # default gamma bound (assume payoffs are positive: true in the limit)
    b = (bet_max - bmins) / (1 + bmins)
    gamma = 2*(b - jnp.log(1+b)) / (b**2) if gamma is None else jnp.broadcast_to(gamma, shape=n)

    def step(carry, xs):
        bet, a = carry
        v, y, bmin, bmax = xs
        g = - v / (1 + bet*v)
        a += g**2
        bet = jnp.clip(bet - g / (y * a), min=bmin, max=bmax)
        return (bet, a), bet

    _, bets = jax.lax.scan(step, (0., a0), (payoffs, gamma, bmins, bmaxs))
    return jnp.concat([jnp.array([0.]), bets[:-1]], axis=0)



