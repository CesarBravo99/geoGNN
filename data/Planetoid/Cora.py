import os
import os.path as osp
import pickle
import numpy as np
import itertools
import scipy.sparse as sp
import urllib
from collections import namedtuple

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])

class CoraData(object):
    
    def __init__(self, data_root="/notebooks/data/Planetoid/Cora/raw", rebuild=False):
        """Cora

        .data:
            * x: 2708 * 1433 np.ndarray
            * y: 2708 np.ndarray
            * adjacency_dict: dict
            * train_mask: 2708，True，False
            * val_mask: 2708，True，False
            * test_mask: 2708，，True，False

        Args:
        -------
            data_root: string, optional
                : ../data/cora
                : {data_root}/weights.pkl
            rebuild: boolean, optional
        """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "weights.pkl")
        print(save_file)
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            with open(save_file, "rb") as f:
                self._data = pickle.load(f)
        else:
            self._data = self.process_data()
            print("Cached file: {}".format(save_file))
            with open(save_file, "wb") as f:
                pickle.dump(self._data, f)

    @property
    def data(self):
        """x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        print("Process data ...")
        self.filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
        x, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        adjacency_dict = graph
        
        num_nodes = x.shape[0]
        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", len(adjacency_dict))
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency_dict=adjacency_dict,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)

        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out
