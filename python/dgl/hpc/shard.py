"""HPC Shard"""

from abc import ABC, abstractmethod
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from .._ffi.ndarray import empty
from typing import Tuple, Callable, Type
from dgl import backend as F

__all__ = ['ShardPolicy', 'ModuloPolicy', 'Shard', 'createTensor', 'TensorShard']

class ShardPolicy(ABC):
  row_size: int
  manager_size: int

  def __init__(self, row_size: int, manager_size: int):
    assert 0 < row_size, "row_size must be greater than 0"
    assert 0 < manager_size, "manager_size must be greater than 0"
    self.row_size = row_size
    self.manager_size = manager_size

  @abstractmethod
  def __getitem__(self, index: int) -> int:
    pass

class ModuloPolicy(ShardPolicy):

  def __init__(self, row_size: int, manager_size: int):
    super().__init__(row_size, manager_size)

  def __getitem__(self, index: int) -> int:
    assert index < self.row_size, "index is out of range"
    assert 0 <= index, "index is out of range"
    return index % self.manager_size

@register_object('hpc.Shard')
class Shard(ObjectBase):

  def __init__(self):
    self.__init_handle_by_constructor__(
      _CAPI_HPCCreateShard
    )

class TensorShard:
  id: int
  name: str
  shape: Tuple[int, ...]
  dtype: Type[F.dtype]
  def __init__(self, id: int, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype]):
    self.id = id
    self.name = name
    self.shape = shape
    self.dtype = dtype

def createTensor(shard: Shard, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype],
  policy: Type[ShardPolicy]) -> TensorShard:
  tensor = empty(shape, F.reverse_data_type_dict[dtype])
  id = _CAPI_HPCRegisterTensor(shard, name, tensor)
  return TensorShard(id, name, shape, dtype)

_init_api("dgl.hpc.shard")
