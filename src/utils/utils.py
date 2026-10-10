import numpy as np
import jax
import jax.numpy as jnp
import torch
import random
import secrets
from typing import Any, Callable
from collections.abc import Sequence, Mapping
from math import floor
from torch.utils.data import Dataset, random_split


def issequence(obj: object, item_type) -> bool:
    return isinstance(obj, Sequence) and all(isinstance(x, item_type) for x in obj)

def isarray(obj: object) -> bool:
    return isinstance(obj, np.ndarray | jnp.ndarray)

def random_seed(nbits=64):
    return secrets.randbits(nbits-1)

def avoid_zero(x: jax.Array, eps: float = 1e-12):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x), eps)

def in_axes(arg2axis: dict[int,int], nargs: int):
    r"""helper function that returns a sequence of values indicating which input array axes to map over.
    Args:
        arg2axis: dictionary whose keys are input array positions and values are the corresponding array axis to map over.
        nargs: total number of input arguments.
    Returns:
        array of size nargs whose value at index is given in arg2axis, and any unspecified index has value None. 
    """
    vargs = nargs*[None]
    for n,ax in arg2axis.items():
        vargs[n] = ax
    return vargs

def split_dataset(dataset: Dataset, split: list, seed: int = 314159):
    r"""partition given dataset into train-val-test splits"""
    assert len(split) == 3
    sizes = [floor(len(dataset)*r/sum(split)) for r in split]
    subsets = random_split(dataset, 
                           lengths=sizes,
                           generator=torch.Generator().manual_seed(seed))
    return {'train': subsets[0], 'val': subsets[1], 'test': subsets[2]}

def stream(dataloader):
    r"""transforms a pytorch dataloader into an infinite generator that continuously loops through the dataset"""
    while True:
        yield from dataloader

def seed_worker(worker_id):
    r"""Set the seed for each dataloader worker. When num_workers > 0, each worker is a copy of the main process with 
    its pytorch seed set to the base_seed + worker_id. However, the random seeds/states of 3rd party libraries is unchanged,
    meaning these worker processes may output the same data!
    To prevent this we can seed the 3rd party libraries with the worker's unique pytorch seed.

    Summary: for reproducibility when num_seeds > 0 (https://docs.pytorch.org/docs/stable/notes/randomness.html)"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_nodes(tree: Mapping, key: Any, value: Any, recursive: bool = True):
    r"""set all the nodes with key to the given value"""
    for k, v in tree.items():
        if key == k:
            tree[k] = value
        elif recursive and isinstance(v, Mapping):
            set_nodes(v, key, value)


