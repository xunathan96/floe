import numpy as np
from typing import Callable
from data import collate, transform
from data.toy.base import ToyDataset, ToyIterator
from data.toy.gaussian import BivariateBimodalGenerator
__all__ = ['TwoSampleGenerator',
           'IndependenceGenerator',]

def split(seed: int | None, num: int = 1):
    if seed is None:
        return [None for i in range(num)]
    rng = np.random.default_rng(seed)
    return rng.integers(2**32, size=num)


# ==============================
#       Two-Sample Problem
# ==============================

class TwoSampleGenerator(ToyIterator):

    def __init__(self,
                 corx: float = 0.5,
                 cory: float = 0.,
                 transform: Callable = transform.identity,
                 seed: int = None):
        subseeds = split(seed, 2)
        self.gmmx = BivariateBimodalGenerator(0., 0., 1., 1., cor1=corx, cor2=-corx, seed=subseeds[0])
        self.gmmy = BivariateBimodalGenerator(0., 0., 1., 1., cor1=cory, cor2=-cory, seed=subseeds[1])
        super().__init__(transform, seed)

    def sample(self):
        x = self.gmmx.sample()
        y = self.gmmy.sample()
        return x, y


# ==============================
#      Independence Problem
# ==============================

class IndependenceGenerator(ToyIterator):

    def __init__(self,
                 cor: float = 0.5,
                 transform: Callable = transform.identity,
                 seed: int = None):
        subseeds = split(seed, 1)
        self.gmm = BivariateBimodalGenerator(0., 0., 1., 1., cor1=cor, cor2=-cor, seed=subseeds[0])
        super().__init__(transform, seed)

    def sample(self):
        u = self.gmm.sample()
        return np.split(u, 2)




