"""HPC Context"""

from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api

__all__ = ['ManagerContext']

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

  @property
  def rank(self) -> int:
    return self.context.rank

  @property
  def size(self) -> int:
    return self.context.size

  def launchWorker(self):
    _CAPI_HPCContextLaunchWorker(self.context)

_init_api("dgl.hpc.context")