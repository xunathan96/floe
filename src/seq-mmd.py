import jax
import jax.numpy as jnp
import argparse
from torch import Generator
from torch.utils.data import DataLoader, IterableDataset
from functools import partial
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from model.kernel.gaussian import gaussian
from statistic.online.mmd import oga_from_matrices, loo_mMMD_from_matrices, online_MMD2_from_matrices, online_var_h1_from_matrices
from testing.betting import universal_portfolio, ons, ons_with_reg, grapa, floe

from data.collate import ComposeCollate, numpy_collate, identity
from utils.experiment import Config, Experiment
from utils.utils import random_seed, stream, set_nodes
from utils.parse import parse_array


def main(args):
    SEED = args.seed
    EXPDIR = args.exp_dir
    EXPID = args.exp_id
    SAVE = args.save
    RETURN_STATS = args.return_stats
    # test settings
    NSTEPS = args.n_steps
    NTESTS = args.n_tests
    LEVEL = args.level
    # critic settings
    BANDWIDTH = jnp.atleast_1d(args.bandwidth)
    SCALE = args.scale
    CRITIC_NORM = 1 / jnp.sqrt(2*SCALE)
    PRED_STRAT = args.pred_strat
    # bet settings
    BET_MIN = args.bet_min
    BET_MAX = args.bet_max
    BET_STRAT = args.bet_strat
    BET_PRIOR = args.bet_prior
    # bet regularization settings
    BET_REG = args.bet_reg
    ALPHA = args.alpha
    DELTA0 = args.delta0
    DELTA1 = args.delta1
    DELTA2 = args.delta2

    cfg = Config(vars(args))
    cfg.parse()
    set_nodes(cfg['dataset'], 'seed', SEED)
    exp = Experiment(EXPDIR)
    exp.build(cfg)

    key = jax.random.key(SEED)
    dataloader = DataLoader(exp['dataset'],
                            batch_size=NSTEPS,
                            shuffle=False if isinstance(exp['dataset'], IterableDataset) else True,
                            drop_last=False if isinstance(exp['dataset'], IterableDataset) else True,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=ComposeCollate([
                                getattr(exp['dataset'], 'collate_hook', False) or identity,
                                numpy_collate,]),
                            generator=Generator().manual_seed(SEED),)
    datastream = stream(dataloader)

    # batch kernel functions
    kernel = partial(gaussian, scale=SCALE)
    kernel = jax.vmap(jax.vmap(jax.vmap(kernel, in_axes=(None,0,None)), in_axes=(0,None,None)), in_axes=(None,None,0))

    def payoff(X, Y):
        Kxx = kernel(X, X, BANDWIDTH)   # (b,n,n)
        Kxy = kernel(X, Y, BANDWIDTH)
        Kyy = kernel(Y, Y, BANDWIDTH)
        if PRED_STRAT == "erm":
            loo_mMMD = jax.vmap(loo_mMMD_from_matrices, in_axes=(0,0,0,None))
            return loo_mMMD(Kxx, Kxy, Kyy, CRITIC_NORM)
        elif PRED_STRAT == "oga":
            oga = jax.vmap(oga_from_matrices, in_axes=(0,0,0,None))
            return oga(Kxx, Kxy, Kyy, CRITIC_NORM)

    def regularized_bets(X, Y):
        Kxx = kernel(X, X, BANDWIDTH)   # (b,n,n)
        Kxy = kernel(X, Y, BANDWIDTH)
        Kyy = kernel(Y, Y, BANDWIDTH)
        zeta = jax.vmap(online_var_h1_from_matrices)(Kxx, Kxy, Kyy) # (b,n)
        mmd2 = jax.vmap(online_MMD2_from_matrices)(Kxx, Kxy, Kyy)   # (b,n)

        n = X.shape[0]
        t = jnp.arange(n) + 1
        mmd2_pen = mmd2
        zeta_pen = jnp.clip(zeta, min=1e-12)
        if DELTA1 is not None:
            eps1 = 2/t + 4*jnp.sqrt(jnp.log(1/DELTA1) / t)
            mmd2_pen = jnp.clip(mmd2 - eps1, min=0)
        if DELTA2 is not None:
            eps2 = 32*jnp.sqrt(2*jnp.log(1/DELTA2) / t)
            zeta_pen = zeta + eps2

        hinv = lambda u: u/3 + jnp.sqrt(2*u)
        num = ALPHA * jnp.sqrt(2*SCALE) * jnp.sqrt(mmd2)
        den = jnp.clip(zeta_pen/4 * hinv(-16 * jnp.log(DELTA0) / (t*zeta_pen)) - mmd2_pen, min=1e-8)
        ceils = jnp.clip(num / den, min=0, max=1)   # (b,n)
        return jnp.pad(ceils[:,:-1], pad_width=[(0,0), (1,0)], constant_values=0.)


    @jax.jit
    def simulate(key, X, Y):
        V_tl = payoff(X, Y)                                  # (b,n,n)
        V_t = jnp.diagonal(V_tl, axis1=-2, axis2=-1)         # (b,n)
        ceils = regularized_bets(X, Y) if BET_REG else None  # (b,n)

        if BET_STRAT == "portfolio":
            f = partial(universal_portfolio, bet_min=BET_MIN, bet_max=BET_MAX, prior=BET_PRIOR, n_portfolios=100)
            bets = jax.vmap(f)(V_t)         # (b,n)
        elif BET_STRAT == "ons":
            f = lambda payoffs, reg_ceils: ons_with_reg(payoffs, bet_min=BET_MIN, bet_max=BET_MAX, reg_ceils=reg_ceils)
            bets = jax.vmap(f)(V_t, ceils)  # (b,n)
        elif BET_STRAT == "grapa":
            f = partial(grapa, bet_min=BET_MIN, bet_max=BET_MAX, bisections=20)
            bets = jax.vmap(f)(V_t)         # (b,n)
        elif BET_STRAT == "floe":
            f = partial(floe, bet_min=BET_MIN, bet_max=BET_MAX, bisections=20)
            bets = jax.vmap(f)(V_tl)        # (b,n)
        elif BET_STRAT == "star":
            n = X.shape[0]
            def star_update(carry, t):
                V0, logW0 = carry   # (b,), (b,)
                nu = (V0 / jnp.clip(t-1, min=1e-2)) + (10*n*jnp.log(8./LEVEL) / jnp.clip(t-1, min=1e-2)**2)
                nu = jnp.clip(nu, max=1.)   # (b,)
                bet1 = jnp.sqrt(2 * jnp.clip(jnp.log(1./LEVEL)-logW0, min=0.) / (nu*(n-t+1)))
                bet1 = jnp.clip(bet1, min=BET_MIN, max=BET_MAX)  # (b,)
                v1 = V_t[:,t-1]   # (b,)
                V1 = V0 + v1**2
                logW1 = logW0 + jnp.log(1 + bet1*v1)
                return (V1, logW1), (bet1, logW1)
            V0 = logW0 = jnp.zeros(V_t.shape[0])
            _, (bets, logW) = jax.lax.scan(star_update, (V0, logW0), jnp.arange(n)+1)
            bets = bets.T   # (b,n)
            logW = logW.T   # (b,n)
            wealth = jnp.exp(logW)
            return {
                "wealth": wealth,   # (b,n)
                "bets": bets,       # (b,n)
                "ceils": ceils,     # (b,n)
            }

        bets = jnp.clip(bets, min=None, max=ceils)  # (b,n)
        logE = jnp.log(1 + bets * V_t)              # (b,n)
        logW = jnp.cumsum(logE, axis=-1)            # (b,n)
        wealth = jnp.exp(logW)
        return {
            "wealth": wealth,   # (b,n)
            "bets": bets,       # (b,n)
            "ceils": ceils,     # (b,n)
        }


    nreject = 0
    info = defaultdict(list)
    pbar = tqdm(total=NTESTS, bar_format="{desc:.80}{percentage:3.0f}%|{bar}{r_bar}")
    for _, (X,Y) in zip(range(NTESTS), datastream):
    
        key, subkey = jax.random.split(key)
        out = simulate(subkey, X, Y)
        wealth = out["wealth"]
        nreject += jnp.max(wealth > 1/LEVEL, axis=-1)

        # update pbar
        b = jnp.argmax(nreject)
        pbar.update()
        pbar.set_description(f"n-reject (b={b}) [{nreject[b]}/{NTESTS}]")

        # store test statistics
        if RETURN_STATS:
            info["wealth"].append(wealth)
            info["bets"].append(out["bets"])
            info["ceils"].append(out["ceils"])

    pbar.close()
    info["power"] = nreject / NTESTS
    return info




def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # configs
    parser.add_argument("--seed", default=random_seed(), type=int)
    parser.add_argument("--exp-dir", default="logs/seq-mmd/", type=str)
    parser.add_argument("--exp-id", type=str)
    parser.add_argument("--save", action="store_true", help="save the config and results to the experiment directory.")
    parser.add_argument("--return-stats", action="store_true", help="also return simulation auxliary values like bets.")
    parser.add_argument("--dataset", type=str, required=True, help="dataset config file.")
    # test params
    parser.add_argument("--n-steps", default=1000, type=int, help="maximum number of steps to run (default=1000).")
    parser.add_argument("--n-tests", default=100, type=int, help="number of tests to average over (default=100).")
    parser.add_argument("--level", default=0.05, type=float, help="significance level (default=0.05).")
    # critic params
    parser.add_argument("--pred-strat", default="erm", choices=["erm", "oga"], help="prediction strategy (default='erm')")
    parser.add_argument("--bandwidth", default=1.0, type=parse_array)
    parser.add_argument("--scale", default=1.0, type=float)
    # bet params
    parser.add_argument("--bet-strat", default="grapa", choices=["ons", "grapa", "floe", "portfolio", "star"], help="betting strategy (default='grapa')")
    parser.add_argument("--bet-min", default=0.0, type=float, help="minimum possible bets (default=0.0)")
    parser.add_argument("--bet-max", default=0.99, type=float, help="maximum possible bets (default=0.99)")
    parser.add_argument("--bet-prior", default="beta", choices=["uniform", "beta"], help="choice of prior for the universal portfolio algorithm (default='beta').")
    # bet regularization
    parser.add_argument("--bet-reg", action="store_true")
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--delta0", default=0.1, type=float)
    parser.add_argument("--delta1", default=0.1, type=float)
    parser.add_argument("--delta2", default=None, type=float)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    jnp.set_printoptions(precision=4, linewidth=10_000)
    print(f"bandwidth: {args.bandwidth}")
    info = main(args)
    print(f"power: {info['power']}")
