import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from abc import abstractmethod
from typing import Callable
from data import transform
__all__ = ['ToyIterator',
           'ToyDataset',]


class ToyIterator(IterableDataset):

    def __init__(self,
                 transform: Callable = transform.identity,
                 seed: int = None,):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.transform = transform
        # NOTE: be careful !! if seed is given, then every instance of ToyGenerator will sample the same sequence

    def __iter__(self):
        return self

    def __next__(self):
        return self.transform(self.sample())

    @abstractmethod
    def sample(self):
        ...


class ToyDataset(Dataset):

    def __init__(self,
                 size: int,
                 transform: Callable = transform.identity,
                 seed: int = 0,):
        super().__init__()
        assert size>0, f"dataset size must be positive, but got {size} instead."
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.transform = transform
        self.data = list()
        for i in range(size):
            self.data.append(self.sample())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])
        return torch.from_numpy(self.data[idx]).float()

    @abstractmethod
    def sample(self):
        ...




