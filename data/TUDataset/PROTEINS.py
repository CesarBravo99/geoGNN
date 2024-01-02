import os
import urllib
from zipfile import ZipFile
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


class DDDataset(object):
    
    def __init__(self, data_root="/notebooks/data/TUDataset/PROTEINS/raw", train_size=0.8):
        self.data_root = data_root
        print(data_root)
        sparse_adjacency, node_labels, graph_indicator, graph_labels = self.read_data()
        self.sparse_adjacency = sparse_adjacency.tocsr()
        self.node_labels = node_labels
        self.graph_indicator = graph_indicator
        self.graph_labels = graph_labels
        self.train_index, self.test_index = self.split_data(train_size)
        self.train_label = graph_labels[self.train_index]
        self.test_label = graph_labels[self.test_index]

    def split_data(self, train_size):
        unique_indicator = np.asarray(list(set(self.graph_indicator)))
        train_index, test_index = train_test_split(unique_indicator,
                                                   train_size=train_size,
                                                   random_state=1234)
        return train_index, test_index
    
    def __getitem__(self, index):
        mask = self.graph_indicator == index
        node_labels = self.node_labels[mask]
        graph_indicator = self.graph_indicator[mask]
        graph_labels = self.graph_labels[index]
        adjacency = self.sparse_adjacency[mask, :][:, mask]
        return adjacency, node_labels, graph_indicator, graph_labels
    
    def __len__(self):
        return len(self.graph_labels)
    
    def read_data(self):
        data_dir = os.path.join(self.data_root, "DD")
        print("Loading DD_A.txt")
        adjacency_list = np.genfromtxt(os.path.join(data_dir, "DD_A.txt"),
                                       dtype=np.int64, delimiter=',') - 1
        print("Loading DD_node_labels.txt")
        node_labels = np.genfromtxt(os.path.join(data_dir, "DD_node_labels.txt"), 
                                    dtype=np.int64) - 1
        print("Loading DD_graph_indicator.txt")
        graph_indicator = np.genfromtxt(os.path.join(data_dir, "DD_graph_indicator.txt"), 
                                        dtype=np.int64) - 1
        print("Loading DD_graph_labels.txt")
        graph_labels = np.genfromtxt(os.path.join(data_dir, "DD_graph_labels.txt"), 
                                     dtype=np.int64) - 1
        num_nodes = len(node_labels)
        sparse_adjacency = sp.coo_matrix((np.ones(len(adjacency_list)), 
                                          (adjacency_list[:, 0], adjacency_list[:, 1])),
                                         shape=(num_nodes, num_nodes), dtype=np.float32)
        print("Number of graphs: ", len(graph_labels))
        print("Number of nodes: ", len(node_labels))
        print("Number of links: ", len(adjacency_list))
        # node_infos = pd.DataFrame(columns=["node_labels", "graph_indicator"])
        # node_infos["node_labels"] = node_labels
        # node_infos["graph_indicator"] = graph_indicator
        return sparse_adjacency, node_labels, graph_indicator, graph_labels
    
