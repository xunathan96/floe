import numpy as np
from typing import Callable
from data.toy.base import ToyDataset, ToyIterator
from data import collate, transform


class BivariateBimodalGenerator(ToyIterator):

    def __init__(self,
                 meanx: float = 0.,
                 meany: float = 0.,
                 varx: float = 1.,
                 vary: float = 1.,
                 *,
                 cor1: float = 0.5,
                 cor2: float = -0.5,
                 ang1: float = None,
                 ang2: float = None,
                 transform: Callable = transform.identity,
                 seed: int = None,):
        r"""Generator class for samples from a bivariate bimodal gaussian with specified means and variances.
        Args:
            cor1: correlation coefficient for the x-axis.
            cor2: correlation coefficient for the y-axis.
            ang1: (optional) angle of displacement (in radians) of the first mixture.
            ang2: (optional) angle of displacement (in radians) of the second mixture."""
        assert 0 < varx and 0 < vary
        assert -1 < cor1 < 1 and -1 < cor2 < 1
        ang1 = np.sign(cor1)*0.25*np.pi if ang1 is None else ang1
        ang2 = np.sign(cor2)*0.25*np.pi if ang2 is None else ang2

        self.mean = np.array([meanx, meany])
        self.std = np.sqrt(np.array([varx, vary]))
        self.cov = np.array([[
            [varx**2, cor*varx*vary],
            [cor*varx*vary, vary**2],
        ] for cor in (cor1, cor2)])
        self.L = np.linalg.cholesky(self.cov)

        # orient correlated axis with x-y axes
        shift1 = -np.sign(cor1)*0.25*np.pi
        shift2 = -np.sign(cor2)*0.25*np.pi
        self.rot = np.array([[
            [np.cos(theta), - np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ] for theta in (ang1+shift1, ang2+shift2)])
        super().__init__(transform, seed)


    def sample(self):
        eps = self.rng.normal(size=(2,2))
        z = self.mean + np.einsum('mij,mj->mi', self.L, eps)    # (mix, 2)
        z = np.einsum('mij,mj->mi', self.rot, z)
        return z[self.rng.choice(2)]    # uniformly sample from one mode



class BivariateBimodalDataset(ToyDataset):

    def __init__(self,
                 size: int,
                 meanx: float = 0.,
                 meany: float = 0.,
                 varx: float = 1.,
                 vary: float = 1.,
                 *,
                 cor1: float = 0.5,
                 cor2: float = -0.5,
                 ang1: float = None,
                 ang2: float = None,
                 transform: Callable = transform.identity,
                 seed: int = 0):
        r"""Dataset class for samples from a bivariate bimodal gaussian with specified means and variances.
        Args:
            cor1: correlation coefficient for the x-axis.
            cor2: correlation coefficient for the y-axis.
            ang1: (optional) angle of displacement (in radians) of the first mixture.
            ang2: (optional) angle of displacement (in radians) of the second mixture."""
        assert 0 < varx and 0 < vary
        assert -1 < cor1 < 1 and -1 < cor2 < 1
        ang1 = np.sign(cor1)*0.25*np.pi if ang1 is None else ang1
        ang2 = np.sign(cor2)*0.25*np.pi if ang2 is None else ang2

        self.mean = np.array([meanx, meany])
        self.std = np.sqrt(np.array([varx, vary]))
        self.cov = np.array([[
            [varx**2, cor*varx*vary],
            [cor*varx*vary, vary**2],
        ] for cor in (cor1, cor2)])
        self.L = np.linalg.cholesky(self.cov)

        # orient correlated axis with x-y axes
        shift1 = -np.sign(cor1)*0.25*np.pi
        shift2 = -np.sign(cor2)*0.25*np.pi
        self.rot = np.array([[
            [np.cos(theta), - np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ] for theta in (ang1+shift1, ang2+shift2)])
        super().__init__(size, transform, seed)


    def sample(self):
        eps = self.rng.normal(size=(2,2))
        z = self.mean + np.einsum('mij,mj->mi', self.L, eps)    # (mix, 2)
        z = np.einsum('mij,mj->mi', self.rot, z)
        return z[self.rng.choice(2)]    # uniformly sample from one mode

