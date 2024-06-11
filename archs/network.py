import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GATv2Conv

class GAT(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads):
        super().__init__()
        self.out_heads = 1

        self.conv1 = GATv2Conv(num_node_features, embedding_size, edge_dim=3, heads=heads, concat=True)
        self.conv2 = GATv2Conv(-1, embedding_size, edge_dim=3, heads=self.out_heads, concat=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)     
        h = F.leaky_relu(h)

        h = self.conv2(x=h, edge_index=edge_index, edge_attr=edge_attr)     

        return h

class Transform(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads):
        super().__init__()
        self.out_heads = 1

        self.conv1 = TransformerConv(num_node_features, embedding_size, edge_dim=2, heads=heads, concat=True) #adjust 2 or 3 for relations
        self.conv2 = TransformerConv(-1, embedding_size, edge_dim=2, heads=self.out_heads, concat=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)     
        h = F.leaky_relu(h)

        h = self.conv2(x=h, edge_index=edge_index, edge_attr=edge_attr)     

        return h