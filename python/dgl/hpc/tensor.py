"""HPC Tensor"""

from dgl import backend as F
from .context import ManagerContext
from .shard import ShardPolicy
from typing import Tuple, Callable, Type

__all__ = ['createTensor', 'TensorShard']

class TensorShard:
  def __init__(self):
    pass

def createTensor(mcontext: ManagerContext, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype],
  policy: Type[ShardPolicy]) -> TensorShard:
  print('createTensor called')
  print('shape', shape)
  return TensorShard()