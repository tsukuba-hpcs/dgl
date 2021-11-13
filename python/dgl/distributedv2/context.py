from mpi4py import MPI
from ctypes import c_ubyte, c_void_p, POINTER, cast
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api

__all__ = [
    'Context',
]


@register_object('ucx.Context')
class Context(ObjectBase):
    def __init__(self, comm = MPI.COMM_WORLD):
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.__init_handle_by_constructor__(
            _CAPI_UCXCreateContext,
            rank,
            size
        )
        addrs = self.__gather_workeraddr(comm)
        print(addrs)
        self.__create_endpoints(addrs)


    def __gather_workeraddr(self, comm):
        # exchange worker address
        addr= _CAPI_UCXGetWorkerAddr(self)
        addrlen = _CAPI_UCXGetWorkerAddrlen(self)
        UByteArr = c_ubyte * addrlen
        UByteArrPtr = POINTER(UByteArr)
        addr = cast(addr, UByteArrPtr)
        addr = bytearray(addr.contents)
        s_msg = [addr, addrlen, MPI.BYTE]
        addrs = bytearray(addrlen * comm.Get_size())
        r_msg = [addrs, addrlen, MPI.BYTE]
        comm.Allgather(s_msg, r_msg)
        return addrs

    def __create_endpoints(self, addrs):
        _CAPI_UCXCreateEndpoints(self, addrs)

    def __del__(self):
        _CAPI_UCXFinalizeContext(self)


_init_api("dgl.distributedv2.context")