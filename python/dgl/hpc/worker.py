from .context import Context
from .._ffi.object import register_object, ObjectBase
from traceback import print_tb
from .._ffi.function import _init_api

__all__ = ['WorkerContext', 'ShardClient']

class WorkerContext(Context):
  """
  DGL's HPC WorkerContext.
  """
  def __enter__(self):
    super().__enter__()
    _CAPI_HPCWorkerConnect(self)
    return self

  def __exit__(self, type, value, traceback):
    super().__exit__(type, value, traceback)
    print('WorkerContext exit with', type, value)

@register_object('hpc.ShardClient')
class ShardClient(ObjectBase):
  def __init__(self, wcontext: WorkerContext):
    self._wcontext = wcontext

  def __enter__(self):
    self.__init_handle_by_constructor__(
      _CAPI_HPCCreateShardClient
    )

  def __exit__(self, type, value, traceback):
   _CAPI_HPCFinalizeShardClient(self)
   print('ShardClient exit with', type, value)
   print_tb(traceback)

_init_api("dgl.hpc.worker")
