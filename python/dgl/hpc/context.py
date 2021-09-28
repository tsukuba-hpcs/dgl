"""HPC Context"""

from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from traceback import print_tb

__all__ = ['Context']

@register_object('hpc.Context')
class Context(ObjectBase):
    """
    DGL's HPC Context.
    """

    @property
    def rank(self) -> int:
        return _CAPI_HPCContextGetRank(self)

    @rank.setter
    def rank(self, value):
        raise ValueError(value, "Reassignment is not allowed")

    @property
    def size(self) -> int:
        return _CAPI_HPCContextGetSize(self)

    def __enter__(self):
        self.__init_handle_by_constructor__(
            _CAPI_HPCCreateContext
        )

    def __exit__(self, type, value, traceback):
        _CAPI_HPCFinalizeContext(self)


_init_api("dgl.hpc.context")
