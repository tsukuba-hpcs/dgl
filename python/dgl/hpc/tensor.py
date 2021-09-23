"""HPC Tensor"""

from dgl import backend as F
from .context import ManagerContext
from .partition import PartitionPolicy
from typing import Tuple, Callable, Type

__all__ = ['serveTensor']

def serveTensor(mcontext: ManagerContext, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype],
  policy: Type[PartitionPolicy], init_func: Callable[[Tuple[int, ...], Type[F.dtype]], None]):
  print('serveTensor called')
  print('shape', shape)