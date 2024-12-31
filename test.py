from graph.kemp_dataset import KempGraphDataset
from graph.kemp_build import get_graph, prune_graph

data = KempGraphDataset(root='data/uniform_seed42', number_of_graphs=5000, need_probs=None, transform=None, pre_transform=None, prune=False, seed=42)
g = data[0]

pruned = prune_graph(g, g.ego_node_idx)
print(pruned)
print(pruned.x)
print(pruned.edge_index)
print(pruned.edge_attr)

print(g.is_undirected())