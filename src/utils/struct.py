import collections.abc as abc
import torch.utils.data as torchdata
from typing import Iterable


class Tagged(abc.MutableMapping):
    r"""wrapper for any object"""

    def __init__(self, tag, obj):
        self.tag = tag
        self.obj = obj

    def __repr__(self):
        return f"({self.tag}){self.obj}"

    def __getitem__(self, key):
        return self.obj[key]
    
    def __setitem__(self, key, value):
        self.obj[key] = value

    def __delitem__(self, key):
        del self.obj[key]

    def __iter__(self):
        return iter(self.obj)
    
    def __len__(self):
        return len(self.obj)

    def items(self):
        if isinstance(self.obj, abc.Mapping):
            return self.obj.items()
        else:
            return []


class StreamData(torchdata.IterableDataset):
    def __init__(self, iterator: Iterable):
        self.iterator = iterator
    def __iter__(self):
        return self.iterator
    # https://stackoverflow.com/questions/73228139/how-to-convert-a-generator-to-a-pytorch-dataloader

