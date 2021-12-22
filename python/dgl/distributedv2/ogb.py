import numpy as np
from os import path
import pandas as pd
import io

__all__ = [
    'node_pred',
    'csv2mmap'
]

# from https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/master.csv
ogb_meta = """
,ogbn-proteins,ogbn-products,ogbn-arxiv,ogbn-mag,ogbn-papers100M
num tasks,112,1,1,1,1
num classes,2,47,40,349,172
eval metric,rocauc,acc,acc,acc,acc
task type,binary classification,multiclass classification,multiclass classification,multiclass classification,multiclass classification
download_name,proteins,products,arxiv,mag,papers100M-bin
version,1,1,1,2,1
url,http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip,http://snap.stanford.edu/ogb/data/nodeproppred/products.zip,http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip,http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip,http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
add_inverse_edge,True,True,False,False,False
has_node_attr,False,True,True,True,True
has_edge_attr,True,False,False,False,False
split,species,sales_ranking,time,time,time
additional node files,node_species,None,node_year,node_year,node_year
additional edge files,None,None,None,edge_reltype,None
is hetero,False,False,False,True,False
binary,False,False,False,False,True
"""

def node_pred(base_path: str, name: str):
    master = pd.read_csv(io.StringIO(ogb_meta), index_col = 0)
    if not name in master:
        raise ValueError("invalid dataset name: {}".format(name))
    meta_info = master[name]
    if int(meta_info['num tasks']) != 1:
        raise ValueError("num tasks must be 1")
    if meta_info['is hetero'] != "False":
        raise ValueError("must be HomoGraph")

    num_node = np.memmap(path.join(base_path, 'raw/num-node-list.dat'), mode='r', dtype='int64')
    assert num_node.shape == (1,), "num_node shape must be (,)"
    num_node = num_node[0]

    num_edge = np.memmap(path.join(base_path, 'raw/num-edge-list.dat'), mode='r', dtype='int64')
    assert num_edge.shape == (1,), "num_edge shape must be (,)"
    num_edge = num_edge[0]

    edge = np.memmap(path.join(base_path, 'raw/edge.dat'), dtype='int64', mode='r')
    edge = edge.reshape((num_edge, 2))
    assert edge.shape == (num_edge, 2), "edge shape must be ({}, 2)".format(num_edge)

    node_feat = np.memmap(path.join(base_path, 'raw/node-feat.dat'), dtype='float32', mode='r')
    node_feat = node_feat.reshape((num_node, node_feat.shape[0] // num_node))
    assert node_feat.shape[0] == num_node, "node feat shape[0] must be {}".format(num_node)

    node_label =  np.memmap(path.join(base_path, 'raw/node-label.dat'), dtype='int16', mode='r')
    assert node_label.shape == num_node, "node label shape must be {}".format(num_node)

    train_nid = np.memmap(path.join(base_path, 'split/{}/train.dat'.format(meta_info['split'])), dtype='int64', mode='r')
    assert node_label.shape == num_node, "node label shape must be {}".format(num_node)

    return int(meta_info['num classes']), edge, node_feat, node_label, train_nid


def csv2mmap(base_path: str, name: str):
    master = pd.read_csv(io.StringIO(ogb_meta), index_col = 0)
    if not name in master:
        raise ValueError("invalid dataset name: {}".format(name))
    meta_info = master[name]
    if int(meta_info['num tasks']) != 1:
        raise ValueError("num tasks must be 1")
    if meta_info['is hetero'] != "False":
        raise ValueError("must be HomoGraph")
    if meta_info['binary'] == 'True':
        raise ValueError("must be CSV")

    # num_node
    num_node = np.genfromtxt(path.join(base_path, 'raw/num-node-list.csv'), delimiter=',', dtype=np.int64)
    assert num_node.shape == (), "num_node shape must be (,)"
    num_node_fp = np.memmap(path.join(base_path, 'raw/num-node-list.dat'), mode='w+', dtype='int64', shape=(1,))
    num_node_fp[:] = num_node

    # num_edge
    num_edge = np.genfromtxt(path.join(base_path, 'raw/num-edge-list.csv'), delimiter=',', dtype=np.int64)
    assert num_edge.shape == (), "num_edge shape must be (,)"
    num_edge_fp = np.memmap(path.join(base_path, 'raw/num-edge-list.dat'), mode='w+', dtype='int64', shape=(1,))
    num_edge_fp[:] = num_edge

    # edge
    edge = np.genfromtxt(path.join(base_path, 'raw/edge.csv'), delimiter=',', dtype=np.int64)
    assert edge.shape == (num_edge, 2), "edge shape must be ({}, 2)".format(num_edge)
    edge_fp = np.memmap(path.join(base_path, 'raw/edge.dat'), mode='w+', dtype='int64', shape=edge.shape)
    edge_fp[:] = edge

    # node_feat
    node_feat = np.genfromtxt(path.join(base_path, 'raw/node-feat.csv'), delimiter=',', dtype=np.float32)
    assert node_feat.shape[0] == num_node, "node feat shape[0] must be {}".format(num_node)
    node_feat_fp = np.memmap(path.join(base_path, 'raw/node-feat.dat'), mode='w+', dtype='float32', shape=node_feat.shape)
    node_feat_fp[:] = node_feat

    # node_label
    node_label = np.genfromtxt(path.join(base_path, 'raw/node-label.csv'), delimiter=',', dtype=np.int16)
    assert node_label.shape == num_node, "node label shape must be {}".format(num_node)
    node_label_fp = np.memmap(path.join(base_path, 'raw/node-label.dat'),  mode='w+', dtype='int16', shape=node_label.shape)
    node_label_fp[:] = node_label

    # train_nid
    train_nid = np.genfromtxt(path.join(base_path, 'split/{}/train.csv'.format(meta_info['split'])), delimiter=',', dtype=np.int64)
    train_nid_fp = np.memmap(path.join(base_path, 'split/{}/train.dat'.format(meta_info['split'])), mode='w+', dtype='int64', shape=train_nid.shape)
    train_nid_fp[:] = train_nid
