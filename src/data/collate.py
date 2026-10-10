import numpy as np
import jax
from torch.utils.data import default_collate
from typing import Callable, Sequence


class ComposeCollate:
    def __init__(self, collates: Sequence[Callable]):
        self.collates = tuple(collates)
    def __repr__(self):
        return f"{self.collates}"
    def __call__(self, batch):
        for collate in self.collates:
            batch = collate(batch)
        return batch


def identity(batch):
    return batch

def collate_fn(batch):
    r"""
    General collate function that handles collection type of element within each batch.

    Args:
        batch: a list of individual samples to be collated into a batch
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
            If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.
    """

def numpy_collate(batch):
    # pytorch default collate, but transform all 'leaves' from torch.Tensors to numpy arrays
    return jax.tree.map(np.asarray, default_collate(batch))


def unzip_pairs(batch: list):
    r""" unzip paired samples """
    # TODO: handle numpy and pytorch data?
    return [np.stack(data) for data in zip(*batch)]

