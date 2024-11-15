import torch
from torch_geometric.data import Data

def get_graph():
    # Define nodes and their labels
    nodes = [
        'MM', 'MF', 'MZy', 'MBy', 'M', 'MZe', 'MBe',
        'FM', 'FF', 'FZy', 'FBy', 'F', 'FZe', 'FBe',
        'Zy', 'By', 'Ego', 'Ze', 'Be', 'ZyD', 'ZyS',
        'ByD', 'ByS', 'D', 'S', 'ZeD', 'ZeS', 'BeD', 'BeS',
        'DD', 'DS', 'SD', 'SS'
    ]
    # Map nodes to indices
    node_map = {node: idx for idx, node in enumerate(nodes)}
    # Define edges (parent-child relationships) and edge attributes
    edges = [
        ('MM', 'MZy'), ('MM', 'MBy'), ('MM', 'M'), ('MM', 'MZe'), ('MM', 'MBe'),
        ('MF', 'MZy'), ('MF', 'MBy'), ('MF', 'M'), ('MF', 'MZe'), ('MF', 'MBe'),
        ('FM', 'FZy'), ('FM', 'FBy'), ('FM', 'F'), ('FM', 'FZe'), ('FM', 'FBe'),
        ('FF', 'FZy'), ('FF', 'FBy'), ('FF', 'F'), ('FF', 'FZe'), ('FF', 'FBe'),
        ('M', 'Ego'), ('M', 'Ze'), ('M', 'Be'), ('M', 'Zy'), ('M', 'By'),
        ('F', 'Ego'), ('F', 'Ze'), ('F', 'Be'), ('F', 'Zy'), ('F', 'By'),
        ('Ego', 'D'), ('Ego', 'S'),
        ('Zy', 'ZyD'), ('Zy', 'ZyS'), ('By', 'ByD'), ('By', 'ByS'),
        ('Ze', 'ZeD'), ('Ze', 'ZeS'), ('Be', 'BeD'), ('Be', 'BeS'),
        ('D', 'DD'), ('D', 'DS'), ('S', 'SD'), ('S', 'SS')
    ]
    # Define initial characteristics for each node: [Gender, Age, SameSex]
    # We only register Gender (0: male, 1: female), Age and SameSex are set according to the ego node
    characteristics = {
        'MM': [1, 0, 0], 'MF': [0, 0, 0], 'MZy': [1, 0, 0], 'MBy': [0, 0, 0], 'M': [1, 0, 0], 'MZe': [1, 0, 0], 'MBe': [0, 0, 0],
        'FM': [1, 0, 0], 'FF': [0, 0, 0], 'FZy': [1, 0, 0], 'FBy': [0, 0, 0], 'F': [0, 0, 0], 'FZe': [1, 0, 0], 'FBe': [0, 0, 0],
        'Zy': [1, 0, 0], 'By': [1, 0, 0], 'Ego': [0, 0, 0], 'Ze': [1, 0, 0], 'Be': [0, 0, 0],
        'ZyD': [1, 0, 0], 'ZyS': [0, 0, 0], 'ByD': [1, 0, 0], 'ByS': [0, 0, 0], 'ZeD': [1, 0, 0], 'ZeS': [0, 0, 0], 'BeD': [1, 0, 0], 'BeS': [0, 0, 0],
        'D': [1, 0, 0], 'S': [0, 0, 0], 'DD': [1, 0, 0], 'DS': [0, 0, 0], 'SD': [1, 0, 0], 'SS': [0, 0, 0]
    }

    # Convert edges to indices
    edge_index = []
    edge_attr = []

    # Add edges and their corresponding attributes
    for src, dst in edges:
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        edge_index.append([src_idx, dst_idx])
        edge_index.append([dst_idx, src_idx])
        edge_attr.append([1., 0.])  # 'parent-of'
        edge_attr.append([0., 1.])  # 'child-of'

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Convert node characteristics to tensor
    x = torch.tensor([characteristics[node] for node in nodes], dtype=torch.float)

    # Create the PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data, node_map

def prune_graph(data, ego_idx):
    """ prune the graph to its bfs-tree rooting at ego node
    """
    from torch_geometric.utils import to_networkx, from_networkx
    import networkx as nx

    def convert_to_networkx_with_attrs(data):
        # Convert to NetworkX graph
        G = to_networkx(data, to_undirected=False, node_attrs=["x"], edge_attrs=["edge_attr"])

        # If attributes were not included by `to_networkx`, add them manually
        if not G.nodes:
            for i in range(data.num_nodes):
                G.nodes[i]['x'] = data.x[i].tolist()
        if not G.edges:
            for i, (u, v) in enumerate(data.edge_index.T):
                G.edges[u.item(), v.item()]['edge_attr'] = data.edge_attr[i].tolist()

        return G

    def convert_networkx_to_torch_geometric(G):
        # Get the edge_index tensor
        edge_index = torch.tensor(list(G.edges)).t().contiguous()

        # Collect node attributes (assuming each node has the same set of attributes)
        if G.nodes:
            node_attr = [None] * len(G.nodes)
            for nid, attrs in G.nodes(data=True):
                node_attr[nid] = attrs['x']
            x = torch.tensor(node_attr, dtype=torch.float)
        else:
            x = None  # No node attributes available

        # Collect edge attributes (assuming each edge has the same set of attributes)
        if G.edges:
            edge_attr = []
            for _, _, attrs in G.edges(data=True):
                edge_attr.append(attrs['edge_attr'])
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_attr = None  # No edge attributes available

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

    G = convert_to_networkx_with_attrs(data)

    bfs_tree = nx.bfs_tree(G, source=ego_idx)
    for node in bfs_tree.nodes:
        if node in G.nodes:
            bfs_tree.nodes[node].update(G.nodes[node])  # Copy node attributes
    for edge in bfs_tree.edges:
        if edge in G.edges:
            bfs_tree.edges[edge].update(G.edges[edge])  # Copy edge attributes


    pruned_data = convert_networkx_to_torch_geometric(bfs_tree)

    # # sanaty check: make sure that the node attr and edge attr are preserved correctly
    # assert (data.x - pruned_data.x).sum().item() == 0
    # edge_idx = {tuple(data.edge_index[:,i].tolist()):i for i in range(data.edge_index.shape[1])}
    # for peid in range(pruned_data.edge_index.shape[1]):
    #     e = tuple(pruned_data.edge_index[:, peid].tolist())
    #     eid = edge_idx[e]
    #     assert (pruned_data.edge_attr[peid] - data.edge_attr[eid]).sum().item() == 0
    return pruned_data


def update_age(data, _, node_map):
    # Define the age map directly
    age_map = {
        1: ['MM', 'MF', 'FM', 'FF', 'MZy', 'MBy', 'M', 'MZe', 'MBe', 'FZy', 'FBy', 'F', 'FZe', 'FBe', 'Ze', 'Be'],  # Older
        0: ['ZyD', 'ZyS', 'ByD', 'ByS', 'D', 'S', 'ZeD', 'ZeS', 'BeD', 'BeS', 'Zy', 'By', 'DD', 'DS', 'SD', 'SS'],  # Younger
    }

    # Iterate through all nodes to update their age based on the age map
    for age, members in age_map.items():
        for node in members:
            if node in node_map:
                idx = node_map[node]
                data.x[idx][1] = age  # Use the age map directly

def update_sex(data, ego_idx):
    # Get the gender of the ego node
    ego_gender = data.x[ego_idx][0]
    # Iterate through all nodes to update their same sex attribute relative to the new ego
    for idx in range(data.num_nodes):
        if data.x[idx][0] == ego_gender:  # Same gender as ego
            data.x[idx][2] = 1  # Same sex
        else:  # Different gender from ego
            data.x[idx][2] = 0  # Different sex

