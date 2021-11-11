from mpi4py import MPI
from ctypes import *
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
        addr = _CAPI_UCXGetWorkerAddr(self)
        print('addr=', addr)
        len = _CAPI_UCXGetWorkerAddrlen(self)
        print('addrlen=', len)

    def __del__(self):
        _CAPI_UCXFinalizeContext(self)


_init_api("dgl.distributedv2.context")