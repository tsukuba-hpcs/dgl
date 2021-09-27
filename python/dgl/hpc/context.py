"""HPC Context"""

from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from .shard import Shard, ShardClient
from traceback import print_tb

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
    self._launched = False

  @property
  def rank(self) -> int:
    return self._context.rank

  @rank.setter
  def rank(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  @property
  def size(self) -> int:
    return self._context.size

  @size.setter
  def size(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  def __enter__(self):
    self._context = Context()
    return self

  def __exit__(self, type, value, traceback):
    print('ManagerContext exit with', type, value)
    print_tb(traceback)
    del self._context

  def launchWorker(self, num_workers: int=1, py: str = "python", worker: str = "worker.py", *args: str):
    assert not self._launched, "cannot launch worker twice."
    self._launched = True
    _CAPI_HPCManagerLaunchWorker(self._context, num_workers, py, worker, *args)

  def serve(self, shard: Shard):
    assert self._launched, "must launch worker."
    _CAPI_HPCManagerServe(self._context, shard)


class WorkerContext:
  """
  DGL's HPC WorkerContext.
  """
  def __init__(self):
    pass

  def __del__(self):
    pass

  @property
  def rank(self) -> int:
    return self._context.rank

  @rank.setter
  def rank(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  @property
  def size(self) -> int:
    return self._context.size

  @size.setter
  def size(self, value):
    raise ValueError(value, "Reassignment is not allowed")

  def __enter__(self):
    self._context = Context()
    self._client = ShardClient()
    _CAPI_HPCWorkerConnect(self._context, self._client)
    return self

  def __exit__(self, type, value, traceback):
    print('WorkerContext exit with', type, value)
    print_tb(traceback)
    _CAPI_HPCFinalizeShardClient(self._client)
    del self._client
    del self._context

_init_api("dgl.hpc.context")