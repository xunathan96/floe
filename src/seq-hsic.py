import jax
import jax.numpy as jnp
import argparse
from torch import Generator
from torch.utils.data import DataLoader, IterableDataset
from functools import partial
from tqdm import tqdm
from collections import defaultdict

from model.kernel.gaussian import gaussian
from statistic.online.hsic import loo_mHSIC_from_matrices as loo_mHSIC
from testing.betting import ons, grapa, floe, universal_portfolio

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
    BANDWIDTH_X = jnp.atleast_1d(args.bandwidth_x)
    BANDWIDTH_Y = jnp.atleast_1d(args.bandwidth_y)
    SCALE_X = args.scale_x
    SCALE_Y = args.scale_y
    CRITIC_NORM = 1/(2*SCALE_X*SCALE_Y)
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
    kernel_x = partial(gaussian, scale=SCALE_X)
    kernel_y = partial(gaussian, scale=SCALE_Y)
    kernel_x = jax.vmap(jax.vmap(jax.vmap(kernel_x, in_axes=(None,0,None)), in_axes=(0,None,None)), in_axes=(None,None,0))
    kernel_y = jax.vmap(jax.vmap(jax.vmap(kernel_y, in_axes=(None,0,None)), in_axes=(0,None,None)), in_axes=(None,None,0))


    def payoff(X, Y):
        X0, X1 = jnp.split(X, 2)
        Y0, Y1 = jnp.split(Y, 2)
        K00 = kernel_x(X0, X0, BANDWIDTH_X)
        K01 = kernel_x(X0, X1, BANDWIDTH_X)
        K11 = kernel_x(X1, X1, BANDWIDTH_X)
        L00 = kernel_y(Y0, Y0, BANDWIDTH_Y)
        L01 = kernel_y(Y0, Y1, BANDWIDTH_Y)
        L11 = kernel_y(Y1, Y1, BANDWIDTH_Y)
        loo_mHSIC_batched = jax.vmap(jax.vmap(loo_mHSIC,
                                              in_axes=(None,None,None,0,0,0,None)), 
                                              in_axes=(0,0,0,None,None,None,None))
        return loo_mHSIC_batched(K00, K01, K11, L00, L01, L11, CRITIC_NORM)

    @jax.jit
    def simulate(key, X, Y):
        V_tl = payoff(X, Y)                     # (bx,by,n,n)
        V_t = V_tl.diagonal(axis1=-2, axis2=-1) # (bx,by,n)

        if BET_STRAT == "portfolio":
            f = partial(universal_portfolio, bet_min=BET_MIN, bet_max=BET_MAX, prior=BET_PRIOR, n_portfolios=100)
            bets = jax.vmap(jax.vmap(f))(V_t)   # (bx,by,n)
        elif BET_STRAT == "ons":
            f = partial(ons, bet_min=BET_MIN, bet_max=BET_MAX)
            bets = jax.vmap(jax.vmap(f))(V_t)   # (bx,by,n)
        elif BET_STRAT == "grapa":
            f = partial(grapa, bet_min=BET_MIN, bet_max=BET_MAX, bisections=20)
            bets = jax.vmap(jax.vmap(f))(V_t)   # (bx,by,n)
        elif BET_STRAT == "floe":
            f = partial(floe, bet_min=BET_MIN, bet_max=BET_MAX, bisections=20)
            bets = jax.vmap(jax.vmap(f))(V_tl)  # (bx,by,n)

        logE = jnp.log(1 + bets * V_t)
        logW = jnp.cumsum(logE, axis=-1)
        wealth = jnp.exp(logW)
        return {
            "wealth": wealth,   # (bx,by,n)
            "bets": bets,       # (bx,by,n)
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
        imax = jnp.argmax(nreject); by = len(BANDWIDTH_Y)
        i, j = (imax // by).item(), (imax % by).item()
        pbar.update()
        pbar.set_description(f"n-reject (bx={i},by={j}) [{nreject[i,j]}/{NTESTS}]")

        # store test statistics
        if RETURN_STATS:
            info["wealth"].append(wealth)
            info["bets"].append(out["bets"])

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
    parser.add_argument("--return-stats", action="store_true", help="return simulation statistics gathered across all tests.")
    parser.add_argument("--dataset", type=str, required=True, help="dataset config file.")
    # test params
    parser.add_argument("--n-steps", default=1000, type=int, help="maximum number of steps to run (default=1000).")
    parser.add_argument("--n-tests", default=100, type=int, help="number of tests to average over (default=100).")
    parser.add_argument("--level", default=0.05, type=float, help="significance level (default=0.05).")
    # critic params
    parser.add_argument("--bandwidth-x", default=1.0, type=parse_array)
    parser.add_argument("--bandwidth-y", default=1.0, type=parse_array)
    parser.add_argument("--scale-x", default=1.0, type=float)
    parser.add_argument("--scale-y", default=1.0, type=float)
    # bet params
    parser.add_argument("--bet-strat", default="grapa", choices=["ons", "grapa", "floe", "portfolio",], help="betting strategy (default='grapa')")
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


if __name__ == "__main__":
    args = parse_args()
    jnp.set_printoptions(precision=4, linewidth=10_000)
    print(f"bandwidth-x: {args.bandwidth_x}")
    print(f"bandwidth-y: {args.bandwidth_y}")
    info = main(args)
    print(f"power: {info['power']}")
