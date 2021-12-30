from ctypes import c_ubyte, c_void_p, POINTER, cast
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api

__all__ = [
    'Communicator',
    'allgather',
]

def allgather(comm, addr, addrlen):
    from mpi4py import MPI
    UByteArr = c_ubyte * addrlen
    UByteArrPtr = POINTER(UByteArr)
    addr = cast(addr, UByteArrPtr)
    addr = bytearray(addr.contents)
    s_msg = [addr, addrlen, MPI.BYTE]
    addrs = bytearray(addrlen * comm.Get_size())
    r_msg = [addrs, addrlen, MPI.BYTE]
    comm.Allgather(s_msg, r_msg)
    return addrs

@register_object('distributedv2.Communicator')
class Communicator(ObjectBase):
    def __init__(self, comm):
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.__init_handle_by_constructor__(
            _CAPI_DistV2CreateCommunicator,
            self.rank,
            self.size
        )

    def create_endpoints(self, comm):
        addr, addrlen = _CAPI_DistV2CreateWorker(self)
        worker_addrs = allgather(comm, addr, addrlen)
        _CAPI_DistV2CreateEndpoints(self, worker_addrs)

_init_api("dgl.distributedv2", __name__)