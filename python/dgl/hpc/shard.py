"""HPC Shard"""

from abc import ABC, abstractmethod
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from .._ffi.ndarray import empty
from typing import Tuple, Callable, Type
from dgl import backend as F

__all__ = ['ShardPolicy', 'ModuloPolicy', 'Shard', 'ShardClient', 'createTensor', 'TensorShard']

class ShardPolicy(ABC):
  row_size: int
  manager_size: int

  def __init__(self, row_size: int, manager_size: int):
    assert 0 < row_size, "row_size must be greater than 0"
    assert 0 < manager_size, "manager_size must be greater than 0"
    self.row_size = row_size
    self.manager_size = manager_size

  @abstractmethod
  def __getitem__(self, index: int) -> Tuple[int, int]:
    pass

  @abstractmethod
  def get_local_row_size(self, rank: int) -> int:
    pass

class ModuloPolicy(ShardPolicy):

  def __init__(self, row_size: int, manager_size: int):
    super().__init__(row_size, manager_size)

  def __getitem__(self, index: int) -> Tuple[int, int]:
    assert index < self.row_size, "index is out of range"
    assert 0 <= index, "index is out of range"
    return (index % self.manager_size, index // self.manager_size)

  def get_local_row_size(self, rank: int) -> int:
    a = self.row_size // self.manager_size
    b = 1 if rank < self.row_size % self.manager_size else 0
    return a + b

@register_object('hpc.Shard')
class Shard(ObjectBase):
  rank: int
  size: int
  def __init__(self, rank: int, size: int):
    self.__init_handle_by_constructor__(
      _CAPI_HPCCreateShard
    )
    self.rank = rank
    self.size = size

@register_object('hpc.ShardClient')
class ShardClient(ObjectBase):
  def __init__(self):
    self.__init_handle_by_constructor__(
      _CAPI_HPCCreateShardClient
    )

  def get_id(self, name: str) -> int:
    return 0

class TensorShard:
  id: int
  name: str
  shape: Tuple[int, ...]
  dtype: Type[F.dtype]
  policy: ShardPolicy
  def __init__(self, id: int, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype],
    policy: ShardPolicy):
    self.id = id
    self.name = name
    self.shape = shape
    self.dtype = dtype
    self.policy = policy

def createTensor(shard: Shard, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype],
  policy: Type[ShardPolicy]) -> TensorShard:
  (row_size, *col_sizes) = shape
  sp: ShardPolicy = policy(row_size, shard.size)
  local_shape = (sp.get_local_row_size(shard.rank), *col_sizes)
  tensor = empty(local_shape, F.reverse_data_type_dict[dtype])
  id = _CAPI_HPCRegisterTensor(shard, name, tensor)
  return TensorShard(id, name, shape, dtype, sp)

_init_api("dgl.hpc.shard")
