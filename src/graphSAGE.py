import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MessagePassing(nn.Module):
    def  __init__(self, feature_dim, agg_dim, upd_dim, 
                  agg_method=lambda x: torch.mean(x, 1), 
                  upd_method=lambda h, m: h + m, 
                  activation=lambda h: F.relu(h)):
        super(MessagePassing, self).__init__()
        self.agg_method = agg_method
        self.upd_method = upd_method
        self.activation = activation
        self.agg_weight = nn.Parameter(torch.Tensor(feature_dim, agg_dim))
        self.upd_weight = nn.Parameter(torch.Tensor(feature_dim, upd_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.agg_weight)
        init.kaiming_uniform_(self.upd_weight)
        
    def forward(self, node_feature, neighborhood_features):
        # Aggregate
        aggr_neighbor = self.agg_method(neighborhood_features)
        message = torch.matmul(aggr_neighbor, self.agg_weight)

        # Update
        node_feature_W = torch.matmul(node_feature, self.upd_weight)
        embedding = self.upd_method(node_feature_W, message)
        return self.activation(embedding)


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(MessagePassing(input_dim, hidden_dim[0], hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(MessagePassing(hidden_dim[index], hidden_dim[index+1], hidden_dim[index+1]))
        self.gcn.append(MessagePassing(hidden_dim[-2], hidden_dim[-1], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
