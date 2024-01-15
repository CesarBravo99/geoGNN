import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool


class basicGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv = torch.nn.ModuleList([])
        self.conv.append(GCNConv(in_channels, hidden_channels))
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv_layer in self.conv:
            x = conv_layer(x, edge_index)
            x = x.relu()
        # x = self.conv1(x, edge_index)
        # x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class basicGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mlp=False, pooling=False):
        super(basicGNN, self).__init__()
        self.mlp = mlp
        self.pooling = pooling
        
        self.conv_layers = torch.nn.ModuleList([GraphConv(in_channels, hidden_channels[0])])
        for i in range(len(hidden_channels)-1):
            self.conv_layers.append(GraphConv(hidden_channels[i], hidden_channels[i+1]))
        if mlp:
            self.final_layer = Linear(hidden_channels[-1], out_channels)
        else:
            self.final_layer = GraphConv(hidden_channels[-1], out_channels)
            
            
    def forward(self, x, edge_index, batch=None):
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
            x = x.relu()
            
        if self.pooling:
            x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        
        if self.mlp:
            x = self.final_layer(x)
        else:
            x = self.final_layer(x, edge_index)
        
        return x