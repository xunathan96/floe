import numpy as np
from itertools import product
from typing import Callable, Sequence
from data.toy.base import ToyDataset, ToyIterator
from data import collate, transform


class BivariateGaussian(ToyIterator):
    def __init__(self,
                 mean1: float = 0.,
                 mean2: float = 0.,
                 eigen1: float = 1.,
                 eigen2: float = 1.,
                 angle: float = 0.,
                 transform = transform.identity,
                 seed = None):
        r"""2-dimensional Gaussian distribution
        Args:
            (mean1, mean2): mean along the x and y coordinates
            (eigen1, eigen2): eigenvalues corresponding to the major and minor axes
            angle: rotation of the principal/major axis relative to the x-axis (degrees)
        """
        super().__init__(transform, seed)
        D = np.array([
            [eigen1, 0],
            [0, eigen2]
        ])
        rad = np.pi*angle/180
        R = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        self.L = R @ np.sqrt(D)
        self.mean = np.array([mean1, mean2])

    def sample(self):
        eps = self.rng.normal(size=2)
        return self.mean + self.L @ eps


class TwoSampleGenerator(ToyIterator):
    def __init__(self,
                 isotropy: float = None,
                 angle: float = None,
                 transform = transform.identity,
                 seed = None):
        r"""Pair of distributions P and Q, where P and Q are each mixtures of 9 Gaussians; P is anisotropic whereas Q is isotropic.
        Args:
            isotropy: Ratio of eigenvalues λ2 / λ1 in (0,1) corresponding to the minor / major axis. If not given, each Gaussian component in P will be randomly assigned a value.
            angle: Rotation of the major axis relative to the x-axis (degrees). If not given, each Gaussian component in P will be randomly assigned a value.
        """
        super().__init__(transform, seed)
        meanx = meany = [-2,0,2]
        if not isinstance(isotropy, Sequence): isotropy = [isotropy for i in range(9)]
        if not isinstance(angle, Sequence): angle = [angle for i in range(9)]
        assert len(isotropy) == len(angle) == 9
        self.p_modes = list()
        self.q_modes = list()

        for (x,y), r, a in zip(product(meanx, meany), isotropy, angle):
            r = self.rng.uniform() if r is None else r      # ratio of minor-major variance
            a = 360*self.rng.uniform() if a is None else a  # principal axis angle
            subseeds = self.rng.integers(2**32, size=2)
            self.p_modes.append(
                BivariateGaussian(mean1=x,
                                  mean2=y,
                                  eigen1=0.25,      # var along principal axis
                                  eigen2=0.25*r,    # var along minor axis
                                  angle=a,
                                  seed=subseeds[0])
            )
            self.q_modes.append(
                BivariateGaussian(mean1=x,
                                  mean2=y,
                                  eigen1=0.25,
                                  eigen2=0.25,
                                  angle=0.,
                                  seed=subseeds[1])
            )

    def sample(self):
        k1, k2 = self.rng.integers(9, size=2)
        x = self.p_modes[k1].sample()
        y = self.q_modes[k2].sample()
        return x,y



