import os
import urllib
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import scipy.sparse as sp
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import torch_scatter
import torch.optim as optim
from utils.utils import *

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = GraphConvolution(input_dim, 1)
    
    def forward(self, adjacency, input_feature, graph_indicator):
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        attn_score = self.activation(attn_score)
        
        mask = top_rank(attn_score, graph_indicator, self.keep_ratio)
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1)
        mask_graph_indicator = graph_indicator[mask]
        mask_adjacency = filter_adjacency(adjacency, mask)
        return hidden, mask_graph_indicator, mask_adjacency
    

class ModelA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """
        Global MODE
        """
        super(ModelA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool = SelfAttentionPooling(hidden_dim * 3, 0.5)
        self.fc1 = nn.Linear(hidden_dim * 3 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, adjacency, input_feature, graph_indicator):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        gcn2 = F.relu(self.gcn2(adjacency, gcn1))
        gcn3 = F.relu(self.gcn3(adjacency, gcn2))
        
        gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim=1)
        pool, pool_graph_indicator, pool_adjacency = self.pool(adjacency, gcn_feature,
                                                               graph_indicator)
        
        readout = torch.cat((global_avg_pool(pool, pool_graph_indicator),
                             global_max_pool(pool, pool_graph_indicator)), dim=1)
        
        fc1 = F.relu(self.fc1(readout))
        fc2 = F.relu(self.fc2(fc1))
        logits = self.fc3(fc2)
        
        return logits


class ModelB(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """
        Hierarchical MODE
        """
        super(ModelB, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.pool1 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool2 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool3 = SelfAttentionPooling(hidden_dim, 0.5)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, num_classes))
    
    def forward(self, adjacency, input_feature, graph_indicator):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        pool1, pool1_graph_indicator, pool1_adjacency = \
            self.pool1(adjacency, gcn1, graph_indicator)
        global_pool1 = torch.cat(
            [global_avg_pool(pool1, pool1_graph_indicator),
             global_max_pool(pool1, pool1_graph_indicator)],
            dim=1)
        
        gcn2 = F.relu(self.gcn2(pool1_adjacency, pool1))
        pool2, pool2_graph_indicator, pool2_adjacency = \
            self.pool2(pool1_adjacency, gcn2, pool1_graph_indicator)
        global_pool2 = torch.cat(
            [global_avg_pool(pool2, pool2_graph_indicator),
             global_max_pool(pool2, pool2_graph_indicator)],
            dim=1)

        gcn3 = F.relu(self.gcn3(pool2_adjacency, pool2))
        pool3, pool3_graph_indicator, pool3_adjacency = \
            self.pool3(pool2_adjacency, gcn3, pool2_graph_indicator)
        global_pool3 = torch.cat(
            [global_avg_pool(pool3, pool3_graph_indicator),
             global_max_pool(pool3, pool3_graph_indicator)],
            dim=1)
        
        readout = global_pool1 + global_pool2 + global_pool3
        
        logits = self.mlp(readout)
        return logits
