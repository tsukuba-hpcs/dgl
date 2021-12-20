from mpi4py import MPI
from ctypes import c_ubyte, c_void_p, POINTER, cast
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api

__all__ = [
    'Communicator',
]


@register_object('distributedv2.Communicator')
class Communicator(ObjectBase):
    def __init__(self, comm = MPI.COMM_WORLD):
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.mpi = comm
        self.__init_handle_by_constructor__(
            _CAPI_DistV2CreateCommunicator,
            self.rank,
            self.size
        )

    def allgather(self, addr, addrlen):
        UByteArr = c_ubyte * addrlen
        UByteArrPtr = POINTER(UByteArr)
        addr = cast(addr, UByteArrPtr)
        addr = bytearray(addr.contents)
        s_msg = [addr, addrlen, MPI.BYTE]
        addrs = bytearray(addrlen * self.size)
        r_msg = [addrs, addrlen, MPI.BYTE]
        self.mpi.Allgather(s_msg, r_msg)
        return addrs

    def create_endpoints(self):
        addr, addrlen = _CAPI_DistV2CreateWorker(self)
        worker_addrs = self.allgather(addr, addrlen)
        _CAPI_DistV2CreateEndpoints(self, worker_addrs)

_init_api("dgl.distributedv2", __name__)