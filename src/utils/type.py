import jax
from collections.abc import Sequence
from typing import Callable, Iterable, Union, Generator, Any
from jaxtyping import ArrayLike

RNGkey = ArrayLike
KernelFn = Callable[[jax.Array, jax.Array], jax.Array]
Axis = Union[int, Sequence[int], None]
DataStream = Generator[Any, None, None]

