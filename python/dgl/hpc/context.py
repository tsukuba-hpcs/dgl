"""HPC Context"""

from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from .shard import Shard, ShardClient

__all__ = ['ManagerContext', 'WorkerContext']

@register_object('hpc.Context')
class Context(ObjectBase):
  """
  DGL's HPC Context.
  """
  def __init__(self):
    self.__init_handle_by_constructor__(
      _CAPI_HPCCreateContext
    )

  def __del__(self):
    _CAPI_HPCFinalizeContext(self)

  @property
  def rank(self) -> int:
    return _CAPI_HPCContextGetRank(self)

  @property
  def size(self) -> int:
    return _CAPI_HPCContextGetSize(self)

class ManagerContext:
  """
  DGL's HPC ManagerContext.
  """
  def __init__(self):
    self.context = Context()
    self.launched = False

  @property
  def rank(self) -> int:
    return self.context.rank

  @rank.setter
  def rank(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  @property
  def size(self) -> int:
    return self.context.size

  @size.setter
  def size(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  def launchWorker(self, num_workers: int=1, py: str = "python", worker: str = "worker.py", *args: str):
    assert not self.launched, "cannot launch worker twice."
    self.launched = True
    _CAPI_HPCManagerLaunchWorker(self.context, num_workers, py, worker, *args)

  def serve(self, shard: Shard):
    assert self.launched, "must launch worker."
    _CAPI_HPCManagerServe(self.context, shard)


class WorkerContext:
  """
  DGL's HPC WorkerContext.
  """
  def __init__(self):
    self.context = Context()
    self.client = ShardClient()
    _CAPI_HPCWorkerConnect(self.context, self.client)

  def __del__(self):
    _CAPI_HPCFinalizeShardClient(self.client)

  @property
  def rank(self) -> int:
    return self.context.rank

  @rank.setter
  def rank(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  @property
  def size(self) -> int:
    return self.context.size

  @size.setter
  def size(self, value):
    raise ValueError(value, "Reassignment is not allowed")

_init_api("dgl.hpc.context")