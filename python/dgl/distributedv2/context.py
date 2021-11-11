from mpi4py import MPI
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api

__all__ = [
    'Context',
]


@register_object('ucx.Context')
class Context(ObjectBase):
    def __init__(self, rank: int, size: int):
        self.__init_handle_by_constructor__(
            _CAPI_UCXCreateContext,
            rank,
            size
        )


_init_api("dgl.distributedv2.context")