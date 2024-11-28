import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATv2Conv, RGCNConv


class GAT(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads):
        super().__init__()
        self.out_heads = 1
        self.n_layers = 3
        edge_dim = 20
        self.f = nn.Linear(num_node_features, embedding_size)
        self.fe = nn.Linear(2, edge_dim)
        self.conv = GATv2Conv(embedding_size, embedding_size, edge_dim=edge_dim, heads=heads, concat=False)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        act_f = F.relu
        h = act_f(self.f(x))
        edge_attr = act_f(self.fe(edge_attr))

        for i in range(self.n_layers):
            h = self.conv(x=h, edge_index=edge_index, edge_attr=edge_attr)
            h = act_f(h)

        return h

class RGCN(nn.Module):
    def __init__(self, num_node_features, embedding_size, num_relations=2): #num_relations=2 for kinship: child-of, parent-of
        super().__init__()
        self.n_layers = 3
        self.embedding_size = embedding_size
        self.num_relations = num_relations

        self.f = nn.Linear(num_node_features, embedding_size)
        self.conv_layers = nn.ModuleList([
            RGCNConv(embedding_size, embedding_size, num_relations=num_relations)
            for _ in range(self.n_layers)
        ])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        act_f = F.relu

        h = act_f(self.f(x))

        edge_type = edge_attr.argmax(dim=1)

        for conv in self.conv_layers:
            h = conv(x=h, edge_index=edge_index, edge_type=edge_type)
            h = act_f(h)

        return h

class Transform(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads):
        super().__init__()
        self.out_heads = 1

        self.conv1 = TransformerConv(num_node_features, embedding_size, edge_dim=2, heads=heads, concat=True) #adjust 2 or 3 for relations
        self.conv2 = TransformerConv(-1, embedding_size, edge_dim=2, heads=self.out_heads, concat=True)
        self.conv3 = TransformerConv(-1, embedding_size, edge_dim=2, heads=self.out_heads, concat=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h = F.leaky_relu(h)

        h = self.conv2(x=h, edge_index=edge_index, edge_attr=edge_attr)
        h = F.leaky_relu(h)

        h = self.conv3(x=h, edge_index=edge_index, edge_attr=edge_attr)

        return h