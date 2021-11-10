import numpy as np
from os import path
import pandas as pd
import io

__all__ = [
    'node_pred'
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

def csv_node_pred(base_path: str, meta_info: dict):
    num_node = np.genfromtxt(path.join(base_path, 'raw/num-node-list.csv'), delimiter=',', dtype=np.int64)
    assert num_node.shape == (), "num_node shape must be (,)"

    num_edge = np.genfromtxt(path.join(base_path, 'raw/num-edge-list.csv'), delimiter=',', dtype=np.int64)
    assert num_edge.shape == (), "num_edge shape must be (,)"

    edge = np.genfromtxt(path.join(base_path, 'raw/edge.csv'), delimiter=',', dtype=np.int64)
    assert edge.shape == (num_edge, 2), "edge shape must be ({}, 2)".format(num_edge)

    node_feat = np.genfromtxt(path.join(base_path, 'raw/node-feat.csv'), delimiter=',', dtype=np.float32)
    assert node_feat.shape[0] == num_node, "node feat shape[0] must be {}".format(num_node)

    node_label = np.genfromtxt(path.join(base_path, 'raw/node-label.csv'), delimiter=',', dtype=np.int8)
    assert node_label.shape == num_node, "node label shape must be {}".format(num_node)

    train_nid = np.genfromtxt(path.join(base_path, 'split/{}/train.csv'.format(meta_info['split'])), delimiter=',', dtype=np.int64)

    return int(meta_info['num classes']), edge, node_feat, node_label, train_nid

def node_pred(base_path: str, name: str):
    master = pd.read_csv(io.StringIO(ogb_meta), index_col = 0)
    if not name in master:
        raise ValueError("invalid dataset name: {}".format(name))
    meta_info = master[name]
    if int(meta_info['num tasks']) != 1:
        raise ValueError("num tasks must be 1")
    if meta_info['is hetero'] != "False":
        raise ValueError("must be HomoGraph")
    binary = meta_info['binary'] == 'True'
    if binary:
        raise NotImplementedError
    else:
        return csv_node_pred(base_path, meta_info)