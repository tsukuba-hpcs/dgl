import numpy as np
import dgl
from array import array
from mpi4py import MPI
from .comm import Communicator
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from ctypes import c_ubyte, c_void_p, POINTER, cast, sizeof
from .. import backend as F
from .. import ndarray as nd

__all__ = [
    'NodeDataLoader'
]

@register_object('distributedv2.NodeDataLoader')
class NodeDataLoader(ObjectBase):
    def __load_temp_edge(self, edges):
        edge_slit = (edges.shape[0] + self.comm.size - 1) // self.comm.size
        l = edge_slit * self.comm.rank
        r = min(l + edge_slit, edges.shape[0])
        return edges[l:r, :]
    def __count_dest_rank(self, edges):
        return np.unique(np.floor_divide(edges[:, 1], self.node_slit), return_counts=True)
    """
    `create_distgraph` creates hetero graph and shard nodes and edges.
    shard policy is node's "Range".
    let nodeX id as x.
    nodeX is managed by rank=⌊x/⌈num_nodes/size⌉⌋ process.
    let edgeAB as a -> b (a is src node id, b is dest node id).
    edgeAB is managed by rank=⌊b/⌈num_nodes/size⌉⌋ process.
    So each process knows incoming edge whose dest node is managed by self.

    It is inefficient for all processes to scan all edges independently.
    Do the following to reduce memory usage and execution times.
    1. each process load temporary edges by load_temp_edge()
    2. each process calculate where to shard temporary edges.
    3. exchange temporary edges by Alltoallv()
    """
    def __create_distgraph(self, edges):
        temp_edges = self.__load_temp_edge(edges)
        vals, cnts = self.__count_dest_rank(temp_edges)
        s_counts = array('Q', [0] * self.comm.size)
        for val, cnt in np.stack((vals, cnts), axis = 1):
            s_counts[val] = cnt
        # r_counts is how many edges this proc will fetch from others.
        r_counts = array('Q', [0] * self.comm.size)
        s_msg = [s_counts, 8, MPI.BYTE]
        r_msg = [r_counts, 8, MPI.BYTE]
        self.comm.mpi.Alltoall(s_msg, r_msg)

        # reuse s_counts as byte counts
        s_counts = [c * (edges.itemsize * 2) for c in s_counts]

        # s_displs is s_counts offset 
        s_displs = array('Q', [0] * self.comm.size)
        for idx in range(1, self.comm.size):
            s_displs[idx] = s_displs[idx-1] + s_counts[idx-1]

        # sort by dest node id.
        temp_edges = temp_edges[temp_edges[:, 1].argsort()]

        # recv_edges is recv buffer.
        recv_edges = np.zeros((sum(r_counts), 2), dtype=edges.dtype)

        # reuse r_counts as byte counts
        r_counts = [c * (edges.itemsize * 2) for c in r_counts]

        # r_displs is r_counts offset
        r_displs = array('Q', [0] * self.comm.size)
        for idx in range(1, self.comm.size):
            r_displs[idx] = r_displs[idx-1] + r_counts[idx-1]

        s_msg = [temp_edges, (s_counts, s_displs), MPI.BYTE]
        r_msg = [recv_edges, (r_counts, r_displs), MPI.BYTE]
        self.comm.mpi.Alltoallv(s_msg, r_msg)

        temp_edges = None

        return dgl.graph_index.from_edge_list((recv_edges[:, 0], recv_edges[:, 1]), True)

    def __load_feats(self, rank, node_slit, feats):
        l = self.node_slit * rank
        r = min(self.node_slit * (rank + 1), feats.shape[0])
        return feats[l:r]

    def __gather_feat_metadata(self):
        rma_id, rkey_buf, rkey_buf_len, addr = _CAPI_DistV2GetFeatMetaData(self)
        # gather rkey buffer
        UByteArr = c_ubyte * rkey_buf_len
        UByteArrPtr = POINTER(UByteArr)
        rkey_buf = cast(rkey_buf, UByteArrPtr)
        rkey_buf = bytearray(rkey_buf.contents)
        s_msg = [rkey_buf, rkey_buf_len, MPI.BYTE]
        rkey_bufs = bytearray(rkey_buf_len * self.comm.size)
        r_msg = [rkey_bufs, rkey_buf_len, MPI.BYTE]
        self.comm.mpi.Allgather(s_msg, r_msg)
        # gather address
        s_msg_addr = [addr, sizeof(addr), MPI.BYTE]
        addrs = bytearray(sizeof(addr) * self.comm.size)
        r_msg_addr = [addrs, sizeof(addr), MPI.BYTE]
        self.comm.mpi.Allgather(s_msg_addr, r_msg_addr)
        return rma_id, rkey_bufs, addrs

    def __del__(self):
        # stop DataLoader, then
        del self.comm

    def __init__(self, dataset, num_layers, edges, feats, labels, max_epoch, fanouts = None, batch_size = 1000, prefetch = 2, seed = 777, comm = MPI.COMM_WORLD):
        self.comm = Communicator(comm)
        self.num_layers = num_layers
        self.labels = labels
        self.max_epoch = max_epoch
        self.prefetch = prefetch
        assert self.prefetch <= self.max_epoch
        self.num_nodes = feats.shape[0]
        print("num_nodes={}".format(self.num_nodes))
        self.node_slit = (self.num_nodes + self.comm.size - 1) // self.comm.size
        if fanouts is not None:
            assert len(fanouts) == self.num_layers
            self.fanouts = fanouts
        else:
            self.fanouts = [30] * self.num_layers
        self.batch_size = batch_size
        self.local_feats = self.__load_feats(self.comm.rank, self.node_slit, feats)
        self.local_graph = self.__create_distgraph(edges)
        print("self.local_graph.number_of_nodes()={}".format(self.local_graph.number_of_nodes()))
        self.epoch = 0
        self.seed = seed
        self.dataset = dataset
        self.num_samples = len(self.dataset) // self.comm.size
        self.total_size = self.num_samples * self.comm.size
        self.__init_handle_by_constructor__(
            _CAPI_DistV2CreateNodeDataLoader,
            self.comm,
            self.num_layers,
            self.num_nodes,
            self.local_graph,
            self.fanouts,
            nd.from_dlpack(F.zerocopy_to_dlpack(F.zerocopy_from_numpy(self.local_feats)))
        )
        rma_id, rkey_bufs, addrs = self.__gather_feat_metadata()
        _CAPI_DistV2SetFeatMetaData(self, rma_id, rkey_bufs, addrs)
        for _ in range(self.prefetch):
            self.__enqueue()

    def __enqueue(self):
        assert self.epoch < self.max_epoch
        g = np.random.default_rng(self.seed + self.epoch)
        indices = g.permutation(len(self.dataset)).tolist()
        indices = indices[:self.total_size]
        indices = indices[self.comm.rank:self.total_size:self.comm.size]
        labels = self.labels[indices]
        length = len(indices)
        for l in range(0, length, self.batch_size):
            r = min(length, l + self.batch_size)
            _CAPI_DistV2EnqueueToNodeDataLoader(self, indices[l:r],
                nd.from_dlpack(F.zerocopy_to_dlpack(F.zerocopy_from_numpy(labels[l:r]))))
        self.num_batches = (length + self.batch_size - 1) // self.batch_size
        self.epoch += 1

    def __iter__(self):
        self.iter = 0
        if self.epoch + self.prefetch < self.max_epoch:
            self.__enqueue()
        return self

    def __next__(self):
        if self.iter < self.num_batches:
            self.iter += 1
            ret = _CAPI_DistV2DequeueToNodeDataLoader(self)
            return ret
        else:
            raise StopIteration

_init_api("dgl.distributedv2", __name__)