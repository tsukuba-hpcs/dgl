"""HPC Shard"""

from abc import ABC, abstractmethod
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from .._ffi.ndarray import empty
from typing import Tuple, Callable, Type
from dgl import backend as F

__all__ = ['ShardPolicy', 'ModuloPolicy', 'Shard', 'ShardClient', 'TensorShard']

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

@register_object('hpc.ShardClient')
class ShardClient(ObjectBase):
  def __init__(self):
    self.__init_handle_by_constructor__(
      _CAPI_HPCCreateShardClient
    )

  def get_id(self, name: str) -> int:
    return 0

class TensorShard:
  _id: int
  _name: str
  _shape: Tuple[int, ...]
  _dtype: Type[F.dtype]
  _local_tensor: F.tensor
  _policy: ShardPolicy
  def __init__(self, id: int, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype],
    local_tensor: F.tensor, policy: ShardPolicy):
    self._id = id
    self._name = name
    self._shape = shape
    self._dtype = dtype
    self._local_tensor = local_tensor
    self._policy = policy

  @property
  def id(self):
    return self._id

  @id.setter
  def id(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  @property
  def name(self):
    return self._name
  
  @name.setter
  def name(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  @property
  def shape(self):
    return self._shape
  
  @shape.setter
  def shape(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  @property
  def dtype(self):
    return self._dtype
  
  @dtype.setter
  def dtype(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  @property
  def local_tensor(self):
    return self._local_tensor

  @local_tensor.setter
  def local_tensor(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  @property
  def policy(self):
    return self._policy

  @policy.setter
  def policy(self, value):
    raise ValueError(value, "Reassignment is not allowed")

@register_object('hpc.Shard')
class Shard(ObjectBase):
  _rank: int
  _size: int
  _tensor: list[F.tensor]

  def __init__(self, rank: int, size: int):
    self.__init_handle_by_constructor__(
      _CAPI_HPCCreateShard
    )
    self._rank = rank
    self._size = size
    self._tensor = []
  
  def createTensor(self, name: str, shape: Tuple[int, ...], dtype: Type[F.dtype],
  policy: Type[ShardPolicy]) -> TensorShard:
    (row_size, *col_sizes) = shape
    sp: ShardPolicy = policy(row_size, self._size)
    local_shape = (sp.get_local_row_size(self._rank), *col_sizes)
    tensor = empty(local_shape, F.reverse_data_type_dict[dtype])
    id = _CAPI_HPCRegisterTensor(self, name, tensor)
    assert len(self._tensor) == id, "number of stored tensor is not equal"
    dlpack = tensor.to_dlpack()
    self._tensor.append(F.zerocopy_from_dlpack(dlpack))
    return TensorShard(id, name, shape, dtype, self._tensor[id], sp)


_init_api("dgl.hpc.shard")
