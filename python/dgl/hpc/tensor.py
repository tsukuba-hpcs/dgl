"""HPC Tensor"""

from dgl import backend as F
from .context import ManagerContext
from .partition import PartitionPolicy
from typing import Tuple, Callable, Type

__all__ = ['serveTensor', 'TensorShard']

class TensorShard:
  def __init__(self):
    pass

def serveTensor(mcontext: ManagerContext, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype],
  policy: Type[PartitionPolicy]) -> TensorShard:
  print('serveTensor called')
  print('shape', shape)
  return TensorShard()