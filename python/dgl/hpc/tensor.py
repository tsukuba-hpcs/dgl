"""HPC Tensor"""

from dgl import backend as F
from .shard import ShardPolicy, Shard
from typing import Tuple, Callable, Type

__all__ = ['createTensor', 'TensorShard']

class TensorShard:
  def __init__(self):
    pass

def createTensor(shard: Shard, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype],
  policy: Type[ShardPolicy]) -> TensorShard:
  print('createTensor called')
  print('shape', shape)
  return TensorShard()