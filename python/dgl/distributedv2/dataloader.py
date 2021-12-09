import numpy as np
import dgl
from array import array
from mpi4py import MPI
from .context import Context
from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api

__all__ = [
    'DistributedSampler',
    'NodeDataLoader'
]


class DistributedSampler:
    def __init__(self, context: Context, dataset, batch_size = 1000, seed = 777):
        self.rank = context.rank
        self.size = context.size
        self.comm = context.comm
        self.context = context
        self.dataset = dataset
        self.epoch = 0
        self.seed = seed
        self.num_samples = len(self.dataset) // self.size
        self.total_size = self.num_samples * self.size
    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        indices = g.permutation(len(self.dataset)).tolist()
        indices = indices[:self.total_size]
        indices = indices[self.rank:self.total_size:self.size]
        self.epoch += 1
        return iter(indices)
    def __len__(self) -> int:
        return self.num_samples

@register_object('distributedv2.NodeDataLoader')
class NodeDataLoader(ObjectBase):
    def __load_temp_edge(self, edges):
        edge_slit = (edges.shape[0] + self.size - 1) // self.size
        l = edge_slit * self.rank
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
        s_counts = array('Q', [0] * self.size)
        for val, cnt in np.stack((vals, cnts), axis = 1):
            s_counts[val] = cnt
        # r_counts is how many edges this proc will fetch from others.
        r_counts = array('Q', [0] * self.size)
        s_msg = [s_counts, 8, MPI.BYTE]
        r_msg = [r_counts, 8, MPI.BYTE]
        self.comm.Alltoall(s_msg, r_msg)

        # reuse s_counts as byte counts
        s_counts = [c * (edges.itemsize * 2) for c in s_counts]

        # s_displs is s_counts offset 
        s_displs = array('Q', [0] * self.size)
        for idx in range(1, self.size):
            s_displs[idx] = s_displs[idx-1] + s_counts[idx-1]

        # sort by dest node id.
        temp_edges = temp_edges[temp_edges[:, 1].argsort()]

        # recv_edges is recv buffer.
        recv_edges = np.zeros((sum(r_counts), 2), dtype=edges.dtype)

        # reuse r_counts as byte counts
        r_counts = [c * (edges.itemsize * 2) for c in r_counts]

        # r_displs is r_counts offset
        r_displs = array('Q', [0] * self.size)
        for idx in range(1, self.size):
            r_displs[idx] = r_displs[idx-1] + r_counts[idx-1]

        s_msg = [temp_edges, (s_counts, s_displs), MPI.BYTE]
        r_msg = [recv_edges, (r_counts, r_displs), MPI.BYTE]
        self.comm.Alltoallv(s_msg, r_msg)

        temp_edges = None

        return dgl.graph((recv_edges[:, 0], recv_edges[:, 1]))

    def __init__(self, sampler: DistributedSampler, num_layers, edges, feats, labels, fanouts = None, batch_size = 1000):
        self.sampler = sampler
        self.rank = self.sampler.rank
        self.size = self.sampler.size
        self.comm = self.sampler.comm
        self.num_layers = num_layers
        self.feats = feats
        self.labels = labels
        self.num_nodes = self.feats.shape[0]
        print("num_nodes={}".format(self.num_nodes))
        self.node_slit = (self.num_nodes + self.size - 1) // self.size
        if fanouts is not None:
            assert len(fanouts) == self.num_layers
            self.fanouts = fanouts
        else:
            self.fanouts = [30] * self.num_layers
        self.batch_size = batch_size
        self.subgraph = self.__create_distgraph(edges)
        print("self.subgraph.num_nodes()={}".format(self.subgraph.num_nodes()))
        self.__init_handle_by_constructor__(
            _CAPI_DistV2CreateNodeDataLoader,
            self.sampler.context,
            self.num_layers,
            self.num_nodes
        )


_init_api("dgl.distributedv2", __name__)