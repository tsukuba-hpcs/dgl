import numpy as np
import dgl
from array import array
from ..heterograph import DGLBlock
from .comm import Communicator, allgather
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from ctypes import c_ubyte, c_void_p, POINTER, cast, sizeof
from .. import backend as F
from .. import ndarray as nd
from collections import deque

__all__ = [
    'EdgeShard',
    'NodeDataLoader',
]

"""
`EdgeShard` creates hetero graph and shard nodes and edges.
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
class EdgeShard:
    MAX_BUFFER_SIZE = 1<<20
    def count_dest_rank(self, edges):
        return np.unique(np.floor_divide(edges[:, 1], self.node_slit), return_counts=True)
    def alltoall(self, comm, temp_edges):
        from mpi4py import MPI
        vals, cnts = self.count_dest_rank(temp_edges)
        s_counts = array('Q', [0] * self.size)
        for val, cnt in np.stack((vals, cnts), axis = 1):
            s_counts[val] = cnt
        # r_counts is how many edges this proc will fetch from others.
        r_counts = array('Q', [0] * self.size)
        s_msg = [s_counts, 8, MPI.BYTE]
        r_msg = [r_counts, 8, MPI.BYTE]
        comm.Alltoall(s_msg, r_msg)
        # reuse s_counts as byte counts
        s_counts = [c * (temp_edges.itemsize * 2) for c in s_counts]
        # s_displs is s_counts offset 
        s_displs = array('Q', [0] * self.size)
        for idx in range(1, self.size):
            s_displs[idx] = s_displs[idx-1] + s_counts[idx-1]
        # sort by dest node id.
        temp_edges = temp_edges[temp_edges[:, 1].argsort()]
        # recv_edges is recv buffer.
        recv_edges = np.zeros((sum(r_counts), 2), dtype=temp_edges.dtype)
        # reuse r_counts as byte counts
        r_counts = [c * (temp_edges.itemsize * 2) for c in r_counts]
        # r_displs is r_counts offset
        r_displs = array('Q', [0] * self.size)
        for idx in range(1, self.size):
            r_displs[idx] = r_displs[idx-1] + r_counts[idx-1]
        s_msg = [temp_edges, (s_counts, s_displs), MPI.BYTE]
        r_msg = [recv_edges, (r_counts, r_displs), MPI.BYTE]
        comm.Alltoallv(s_msg, r_msg)
        return recv_edges

    def __init__(self, comm, edges, node_slit):
        assert edges.shape[1] == 2
        edges_len = edges.shape[0]
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.node_slit = node_slit
        self.edge_slit = (edges_len + self.size - 1) // self.size
        l = self.edge_slit * self.rank
        r = min(l + self.edge_slit, edges_len)
        batched_edges = [
            self.alltoall(comm,
                edges[batch_l:min(batch_l + self.MAX_BUFFER_SIZE, r), :]
            )
            for batch_l in range(l, r, self.MAX_BUFFER_SIZE)
        ]
        shard = np.concatenate(batched_edges)
        shard = shard[np.lexsort((shard[:,0], shard[:,1]))]
        self._src = shard[:, 0]
        self._dst = shard[:, 1]
    @property
    def src(self):
        return self._src
    @property
    def dst(self):
        return self._dst

@register_object('distributedv2.NodeDataLoader')
class NodeDataLoader(ObjectBase):
    def __load_feats(self, rank, node_slit, feats):
        l = self.node_slit * rank
        r = min(self.node_slit * (rank + 1), feats.shape[0])
        return feats[l:r]

    def __gather_feat_metadata(self, comm):
        rkeybuf, rkeybuf_len, addr, addr_len = _CAPI_DistV2MapRMAService(self)
        # gather rkey buffer
        rkeybufs = allgather(comm, rkeybuf, rkeybuf_len)
        # gather address
        addrs = allgather(comm, addr, addr_len)
        return rkeybufs, addrs

    def __del__(self):
        # stop DataLoader, then
        from mpi4py import MPI
        MPI.COMM_WORLD.barrier()
        _CAPI_DistV2TermNodeDataLoader(self)
        del self.comm

    def __setup(self, edges, feats):
        from mpi4py import MPI
        self.world_rank = MPI.COMM_WORLD.Get_rank()
        self.world_size = MPI.COMM_WORLD.Get_size()
        local_comm = None
        if self.procs_per_dataset <= 0:
            local_comm = MPI.COMM_WORLD
        else:
            assert self.world_size >= self.procs_per_dataset
            assert self.world_size % self.procs_per_dataset == 0
            color = self.world_rank // self.procs_per_dataset
            key = self.world_rank % self.procs_per_dataset
            local_comm = MPI.COMM_WORLD.Split(color, key)
        self.comm = Communicator(local_comm)
        self.node_slit = (self.num_nodes + self.comm.size - 1) // self.comm.size
        shard = EdgeShard(local_comm, edges, self.node_slit)
        self.local_feats = self.__load_feats(self.comm.rank, self.node_slit, feats)
        self.num_samples = len(self.dataset) // self.world_size
        self.total_size = self.num_samples * self.world_size
        self.num_batch = (self.num_samples + self.batch_size - 1) // self.batch_size
        assert self.prefetch <= self.num_batch * self.max_epoch
        assert 0 <= self.prefetch
        self.__init_handle_by_constructor__(
            _CAPI_DistV2CreateNodeDataLoader,
            self.comm,
            self.num_layers,
            self.num_nodes,
            self.fanouts,
            F.zerocopy_to_dgl_ndarray(F.zerocopy_from_numpy(self.local_feats)),
            F.zerocopy_to_dgl_ndarray(F.zerocopy_from_numpy(shard.src)),
            F.zerocopy_to_dgl_ndarray(F.zerocopy_from_numpy(shard.dst))
        )
        self.comm.create_endpoints(local_comm)
        rkeybufs, addrs = self.__gather_feat_metadata(local_comm)
        _CAPI_DistV2PrepareRMAService(self, rkeybufs, addrs)

    def __init__(self, dataset, num_layers, edges, feats, labels, max_epoch, fanouts = None, batch_size = 1000, prefetch = 5, seed = 777, procs_per_dataset = 0):
        self.num_layers = num_layers
        self.labels = labels[:]
        self.dataset = dataset[:]
        self.max_epoch = max_epoch
        self.prefetch = prefetch
        self.num_nodes = feats.shape[0]
        if fanouts is not None:
            assert len(fanouts) == self.num_layers
            self.fanouts = fanouts
        else:
            self.fanouts = [-1] * self.num_layers
        self.procs_per_dataset = procs_per_dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.__setup(edges, feats)
        # launch
        _CAPI_DistV2LaunchNodeDataLoader(self)
        self.pre_iter = 0
        self.__reset()
        for _ in range(self.prefetch):
            self.__enqueue()

    def __enqueue(self):
        assert self.pre_iter < self.num_batch * self.max_epoch
        l = self.batch_size * (self.pre_iter % self.num_batch)
        r = min(self.num_samples, l + self.batch_size)
        _CAPI_DistV2EnqueueToNodeDataLoader(self,
                F.zerocopy_to_dgl_ndarray(F.zerocopy_from_numpy(self.indices[l:r])),
                F.zerocopy_to_dgl_ndarray(F.zerocopy_from_numpy(self.iter_labels[l:r])))
        self.pre_iter += 1
        if self.pre_iter % self.num_batch == 0:
            self.__reset()

    def __reset(self):
        g = np.random.default_rng(self.seed + (self.pre_iter // self.num_batch))
        self.indices = self.dataset[g.permutation(len(self.dataset))[self.world_rank:self.total_size:self.world_size]]
        self.iter_labels = self.labels[self.indices]

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter >= self.num_batch:
            raise StopIteration
        if self.pre_iter < self.num_batch * self.max_epoch:
            self.__enqueue()
        self.iter += 1
        _blocks, labels, feats = _CAPI_DistV2DequeueToNodeDataLoader(self)
        blocks = [DGLBlock(block, (['_N'], ['_N'])) for block in _blocks]
        return blocks, F.from_dgl_nd(feats),  F.from_dgl_nd(labels)

_init_api("dgl.distributedv2", __name__)