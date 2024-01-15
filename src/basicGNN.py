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
        
        self.conv = torch.nn.ModuleList([])
        for i in range(len(hidden_channels)-1):
            if i == 0:
                self.conv.append(GCNConv(in_channels, hidden_channels[i]))
            else:
                self.conv.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
        if mlp:
            self.final_layer = Linear(hidden_channels[-1], out_channels)
        else:
            self.final_layer = GCNConv(hidden_channels[-1], out_channels)
            
            
        # self.conv1 = GraphConv(in_channels, hidden_channels)
        # self.conv2 = GraphConv(hidden_channels, hidden_channels)
        # self.conv3 = GraphConv(hidden_channels, hidden_channels)
        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        for conv_layer in self.conv:
            x = conv_layer(x, edge_index)
            x = x.relu()
            
        if self.pooling:
            x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        
        if self.mlp:
            x = self.final_layer(x)
        else:
            x = self.final_layer(x, edge_index)
        
        # 1. Obtain node embeddings 
        # x = self.conv1(x, edge_index)
        # x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)
        
        return x