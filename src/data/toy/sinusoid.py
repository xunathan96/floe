import numpy as np
from typing import Callable
from data.toy.base import ToyDataset, ToyIterator
from data import transform
__all__ = ['TwoSampleGenerator',
           'SparseTwoSampleGenerator',
           'TwoSampleDataset',
           'SparseTwoSampleDataset'
           'IndependenceGenerator',
           'SparseIndependenceGenerator',
           'IndependenceDataset',
           'SparseIndependenceDataset']

# ==============================
#       Two-Sample Problem
# ==============================

class TwoSampleGenerator(ToyIterator):

    def __init__(self,
                 dim: int,
                 frequency: float,
                 amplitude: float,
                 transform = transform.identity, 
                 seed = None):
        r"""Generator class for samples from the Sinusoid distribution (p) and the standard uniform (q)."""
        assert dim>0, f"dimension be positive, but got {dim} instead."
        assert frequency>=0, f"frequency be non-negative, but got {frequency} instead."
        assert amplitude>=0, f"amplitude be non-negative, but got {amplitude} instead."
        self.dim = dim
        self.l = frequency
        self.amp = amplitude
        self.M = 1. + amplitude  # M-factor used in rejection sampling from uniform proposal.
        super().__init__(transform, seed)

    def prob(self, x):
        r"""computes the unnormalized density p(x) = 1 + c * sin(l*x1)...sin(l*xd)"""
        return 1 + self.amp * np.prod(np.sin(self.l * x), axis=-1)

    def sample(self) -> np.ndarray:
        r"""rejection sampling a single observation from p."""
        alpha = 1
        keep_prob = 0
        while alpha >= keep_prob:
            x = self.rng.uniform(low=-np.pi, high=np.pi, size=self.dim)     # (d,)
            y = self.rng.uniform(low=-np.pi, high=np.pi, size=self.dim)     # (d,)
            alpha = self.rng.uniform()
            keep_prob = self.prob(x) / self.M   # p(x)/Mq(x)
        return x, y


class SparseTwoSampleGenerator(TwoSampleGenerator):

    def __init__(self,
                 dim: int,
                 sparsity: int,
                 frequency: float,
                 amplitude: float,
                 transform = transform.identity,
                 seed: int = None):
        assert 0 <= sparsity < dim
        self.sparsity = sparsity 
        super().__init__(dim, frequency, amplitude, transform, seed)

    def sample(self) -> np.ndarray:
        r"""rejection sampling a single observation from p."""
        alpha = 1
        keep_prob = 0
        while alpha >= keep_prob:
            x = self.rng.uniform(low=-np.pi, high=np.pi, size=self.dim)     # (d,)
            y = self.rng.uniform(low=-np.pi, high=np.pi, size=self.dim)     # (d,)
            alpha = self.rng.uniform()
            keep_prob = self.prob(x[:self.dim-self.sparsity]) / self.M      # threshold computed based only on the non-sparse dimensions.
        return x, y


class TwoSampleDataset(ToyDataset):

    def __init__(self,
                 size: int,
                 dim: int,
                 frequency: float,
                 amplitude: float,
                 transform: Callable = transform.identity,
                 seed: int = 0):
        r"""Dataset class for samples from the Sinusoid distribution (p) and the standard uniform (q)."""
        assert dim>0, f"dimension be positive, but got {dim} instead."
        assert frequency>=0, f"frequency be non-negative, but got {frequency} instead."
        assert amplitude>=0, f"amplitude be non-negative, but got {amplitude} instead."
        self.dim = dim
        self.l = frequency
        self.amp = amplitude
        self.M = 1. + amplitude  # M-factor used in rejection sampling from uniform proposal.
        super().__init__(size, transform, seed)

    def prob(self, x):
        r"""computes the unnormalized density p(x) = 1 + c^d*sin(l*x1)...sin(l*xd)"""
        return 1 + self.amp * np.prod(np.sin(self.l * x), axis=-1)

    def sample(self):
        r"""rejection sampling a single observation from p."""
        alpha = 1
        keep_prob = 0
        while alpha >= keep_prob:
            x = self.rng.uniform(low=-np.pi, high=np.pi, size=self.dim)     # (d,)
            y = self.rng.uniform(low=-np.pi, high=np.pi, size=self.dim)     # (d,)
            alpha = self.rng.uniform()
            keep_prob = self.prob(x) / self.M
        return x, y


class SparseTwoSampleDataset(TwoSampleDataset):

    def __init__(self,
                 size: int,
                 dim: int,
                 sparsity: int,
                 frequency: float,
                 amplitude: float,
                 transform = transform.identity,
                 seed: int = 0):
        assert 0 <= sparsity < dim
        self.sparsity = sparsity 
        super().__init__(size, dim, frequency, amplitude, transform, seed)

    def sample(self):
        r"""rejection sampling a single observation from p."""
        alpha = 1
        keep_prob = 0
        while alpha >= keep_prob:
            x = self.rng.uniform(low=-np.pi, high=np.pi, size=self.dim)     # (d,)
            y = self.rng.uniform(low=-np.pi, high=np.pi, size=self.dim)     # (d,)
            alpha = self.rng.uniform()
            keep_prob = self.prob(x[:self.dim-self.sparsity]) / self.M      # threshold computed based only on the non-sparse dimensions.
        return x, y


# ==============================
#      Independence Problem
# ==============================

class IndependenceGenerator(ToyIterator):

    def __init__(self,
                 frequency: float,
                 amplitude: float,
                 dim: int = 1,
                 transform = transform.identity, 
                 seed = None):
        r"""Generator class for samples from the Sinusoid distribution"""
        assert dim>0, f"dimension must be positive, but got {dim} instead."
        assert frequency>=0, f"frequency must be non-negative, but got {frequency} instead."
        assert amplitude>=0, f"amplitude must be non-negative, but got {amplitude} instead."
        self.dim = dim
        self.l = frequency
        self.amp = amplitude
        self.M = 1. + amplitude  # M-factor used in rejection sampling from uniform proposal.
        super().__init__(transform, seed)

    def prob(self, x):
        r"""computes the unnormalized density p(x) = 1 + c * sin(l*x1)...sin(l*xd)"""
        return 1 + self.amp * np.prod(np.sin(self.l * x))

    def sample(self) -> np.ndarray:
        accept = False
        while not accept:
            u = self.rng.uniform(-np.pi, np.pi, size=2*self.dim)    # (2d,)
            thresh = self.prob(u) / self.M
            if self.rng.uniform() <= thresh:
                accept = True
        return np.split(u, 2)


class SparseIndependenceGenerator(IndependenceGenerator):

    def __init__(self,
                 dim: int,
                 sparsity: int,
                 frequency: float,
                 amplitude: float,
                 transform = transform.identity,
                 seed: int = None):
        assert 0 <= sparsity < dim
        self.sparsity = sparsity 
        super().__init__(dim, frequency, amplitude, transform, seed)

    def sample(self) -> np.ndarray:
        r"""rejection sampling a single observation from p."""
        alpha = 1
        keep_prob = 0
        while alpha >= keep_prob:
            u = self.rng.uniform(low=-np.pi, high=np.pi, size=2*self.dim)   # (2d,)
            alpha = self.rng.uniform()
            keep_prob = self.prob(u[:2*(self.dim - self.sparsity)]) / self.M        
        return u[::2].copy(), u[::-2].copy()
        # split into X and Y with dependent dimensions paired in opposite directions, i.e. (X1,Yd), ..., (Xd,Y1).


class IndependenceDataset(ToyDataset):

    def __init__(self,
                 size: int,
                 dim: int,
                 frequency: float,
                 amplitude: float,
                 transform: Callable = transform.identity,
                 seed: int = 0):
        r"""Dataset class for dependent samples from the Sinusoid distribution (p)."""
        assert dim>0, f"dimension be positive, but got {dim} instead."
        assert frequency>=0, f"frequency be non-negative, but got {frequency} instead."
        assert amplitude>=0, f"amplitude be non-negative, but got {amplitude} instead."
        self.dim = dim
        self.l = frequency
        self.amp = amplitude
        self.M = 1. + amplitude  # M-factor used in rejection sampling from uniform proposal.
        super().__init__(size, transform, seed)

    def prob(self, x):
        r"""computes the unnormalized density p(x) = 1 + c^d*sin(l*x1)...sin(l*xd)"""
        return 1 + self.amp * np.prod(np.sin(self.l * x), axis=-1)

    def sample(self):
        r"""rejection sampling a single observation from p."""
        alpha = 1
        keep_prob = 0
        while alpha >= keep_prob:
            u = self.rng.uniform(low=-np.pi, high=np.pi, size=2*self.dim)   # (2d,)
            alpha = self.rng.uniform()
            keep_prob = self.prob(u) / self.M   # p(x)/Mq(x)
        return u[:self.dim], u[self.dim:]


class SparseIndependenceDataset(IndependenceDataset):

    def __init__(self,
                 size: int,
                 sparsity: int,
                 dim: int,
                 frequency: float,
                 amplitude: float,
                 transform = transform.identity,
                 seed: int = 0):
        assert 0 <= sparsity < dim
        self.sparsity = sparsity 
        super().__init__(size, dim, frequency, amplitude, transform, seed)

    def sample(self):
        r"""rejection sampling a single observation from p."""
        alpha = 1
        keep_prob = 0
        while alpha >= keep_prob:
            u = self.rng.uniform(low=-np.pi, high=np.pi, size=2*self.dim)   # (2d,)
            alpha = self.rng.uniform()
            keep_prob = self.prob(u[:2*(self.dim - self.sparsity)]) / self.M
        return u[::2].copy(), u[::-2].copy()


