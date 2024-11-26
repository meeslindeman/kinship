import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads):
        super().__init__()
        self.out_heads = 1
        self.n_layers = 3
        self.f = nn.Linear(num_node_features, embedding_size)
        self.conv = GATv2Conv(embedding_size, embedding_size, edge_dim=2, heads=heads, concat=False)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.f(x)

        for i in range(self.n_layers):
            h = self.conv(x=h, edge_index=edge_index, edge_attr=edge_attr)
            h = F.leaky_relu(h)

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