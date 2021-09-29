from .context import Context
from dgl import backend as F
from typing import Tuple, Type
from .._ffi.object import register_object, ObjectBase
from traceback import print_tb
from .._ffi.function import _init_api
import ctypes

__all__ = ['WorkerContext', 'TensorClient', 'ShardClient']

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

class TensorClient:
    _id: int
    _dtype: F.dtype
    _col_shape: Tuple[int, ...]

    def __init__(self, id: int, dtype: F.dtype, col_shape: Tuple[int, ...]):
        self._id = id
        self._dtype = dtype
        self._col_shape = col_shape
    
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        raise ValueError(value, "Reassignment is not allowed")


@register_object('hpc.ShardClient')
class ShardClient(ObjectBase):
    def __init__(self, wcontext: WorkerContext):
        self._wcontext = wcontext

    def __enter__(self):
        self.__init_handle_by_constructor__(
        _CAPI_HPCCreateShardClient
        )
        _CAPI_HPCWorkerRecvMetadata(self._wcontext, self)
        _CAPI_HPCAllocSlicePool(self, 1)
        return self

    def __exit__(self, type, value, traceback):
        _CAPI_HPCFinalizeShardClient(self)
        print('ShardClient exit with', type, value)
        print_tb(traceback)

    def getMetadata(self, name: str) -> TensorClient:
        id = _CAPI_HPCGetTensorIDFromName(self, name)
        dtype = _CAPI_HPCGetTensorDtypeFromID(self, id)
        colshapeList = _CAPI_HPCGetTensorShapeFromID(self, id)
        colshape = tuple(colshapeList)
        return TensorClient(id, F.data_type_dict[dtype], colshape)

    def fetchSlice(self, tc: TensorClient, rank, row) -> F.tensor:
        tensor = _CAPI_HPCFetchSlice(self._wcontext, self, tc.id, rank, row)
        return F.zerocopy_from_dgl_ndarray(tensor)

_init_api("dgl.hpc.worker")
