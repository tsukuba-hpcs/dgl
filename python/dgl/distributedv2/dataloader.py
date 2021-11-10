import numpy as np
import dgl
from array import array
from mpi4py import MPI

__all__ = [
    'RangePolicy',
    'create_distgraph'
]

class RangePolicy:
    def __init__(self, rank: int, size: int, num_nodes: int):
        self._rank = rank
        self._size = size
        self._num_nodes = num_nodes
        self._node_slit = (num_nodes + size - 1) // size
    def load_temp_edge(self, edge):
        edge_slit = (edge.shape[0] + self._size - 1) // self._size
        l = edge_slit * self._rank
        r = min(l + edge_slit, edge.shape[0])
        return edge[l:r, :]
    def count_dest_rank(self, part_edge):
        return np.unique(np.floor_divide(part_edge[:, 1], self._node_slit), return_counts=True)
    def sort_by_dest(self, part_edge):
        return part_edge[part_edge[:, 1].argsort()]
    def is_src_local(self, part_edge):
        return np.floor_divide(part_edge[:, 0], self._node_slit) == self._rank
    @property
    def rank(self):
        return self._rank
    @property
    def size(self):
        return self._size
    @property
    def num_nodes(self):
        return self._num_nodes

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
def create_distgraph(policy, edge):
    temp_edge = policy.load_temp_edge(edge)
    vals, cnts = policy.count_dest_rank(temp_edge)

    # s_counts is how many edges this proc will send to others.
    s_counts = array('Q', [0] * policy.size)
    for val, cnt in np.stack((vals, cnts), axis = 1):
        s_counts[val] = cnt

    comm = MPI.COMM_WORLD

    # r_counts is how many edges this proc will fetch from others.
    r_counts = array('Q', [0] * policy.size)
    s_msg = [s_counts, 8, MPI.BYTE]
    r_msg = [r_counts, 8, MPI.BYTE]
    comm.Alltoall(s_msg, r_msg)

    # reuse s_counts as byte counts
    s_counts = [c * (edge.itemsize * 2) for c in s_counts]

    # s_displs is s_counts offset 
    s_displs = array('Q', [0] * policy.size)
    for idx in range(1, policy.size):
        s_displs[idx] = s_displs[idx-1] + s_counts[idx-1]

    # sort by dest node id.
    temp_edge = temp_edge[temp_edge[:, 1].argsort()]

    # recv_edge is recv buffer.
    recv_edge = np.zeros((sum(r_counts), 2), dtype=edge.dtype)

    # reuse r_counts as byte counts
    r_counts = [c * (edge.itemsize * 2) for c in r_counts]

    # r_displs is r_counts offset
    r_displs = array('Q', [0] * policy.size)
    for idx in range(1, policy.size):
        r_displs[idx] = r_displs[idx-1] + r_counts[idx-1]

    s_msg = [temp_edge, (s_counts, s_displs), MPI.BYTE]
    r_msg = [recv_edge, (r_counts, r_displs), MPI.BYTE]
    comm.Alltoallv(s_msg, r_msg)

    temp_edge = None

    cond = policy.is_src_local(recv_edge)
    local_edge = recv_edge[np.nonzero(cond)]
    bridge_edge = recv_edge[np.nonzero(~cond)]

    return dgl.heterograph({
        ('@local/_V', '_E', '@local/_V'): (local_edge[:,0], local_edge[:,1]),
        ('@remote/_V', '_E', '@local/_V'): (bridge_edge[:,0], bridge_edge[:,1]),
    })
