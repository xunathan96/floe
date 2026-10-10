import numpy as np
import jax.numpy as jnp
import torch
from torchvision.transforms.v2 import *

def identity(input):
    # delete this and use Identity in v2?
    return input

class ToFloat32(ToDtype):
    def __init__(self, scale = False):
        super().__init__(torch.float32, scale)

class Flatten:
    def __call__(self, input):
        return Lambda(torch.flatten)(input)

class FlattenAndCast:
    # depreciated? where is this used?
    def __call__(self, pic):
        return np.ravel(np.asarray(pic, dtype=jnp.float32))



