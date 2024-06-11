import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import to_networkx

def generate_sequence(graph, edges_away: int = 2):
    # Axiom mappings and encoding dictionary
    axioms = {
        'gender': {0: 'Male', 1: 'Female'},
        'age': {0: 'Older', 1: 'Equal', 2: 'Younger'},
        'relationship': {(1, 0, 0): 'Spouse', (0, 1, 0): 'Child', (0, 0, 1): 'Parent'}
    }
    encoding = {'Female': 1, 'Male': 2, 'Older': 3, 'Younger': 4, 'Equal': 5, 'Parent': 6, 'Child': 7, 'Spouse': 8}

    # Extract relevant graph data
    ego_idx = graph.ego_node_idx
    target_idx = graph.target_node_idx
    node_features = graph.x
    edges = graph.edge_index
    edge_attrs = graph.edge_attr

    # Convert graph to NetworkX format for shortest path search
    graphx = to_networkx(graph, to_undirected=True)
    paths = list(nx.all_shortest_paths(graphx, source=ego_idx, target=target_idx))

    if not paths:
        return torch.zeros((1, 4), dtype=torch.float32)  # Return an empty tensor if no valid paths are found

    # Take the first shortest path found
    path = paths[0]

    # Prepare the sequence to store encoded data
    sequence = []

    def get_relationship_type(node_a, node_b):
        for i in range(edges.shape[1]):
            if edges[0, i] == node_a and edges[1, i] == node_b:
                return axioms['relationship'].get(tuple(edge_attrs[i].tolist()), "Unknown")
        return "Unknown"

    # Traverse the path and gather relationships
    for i in range(len(path) - 1):
        node_a = path[i]
        node_b = path[i + 1]

        relationship = get_relationship_type(node_a, node_b)
        if relationship == "Unknown":
            continue

        gender = axioms['gender'][int(node_features[node_b][0])]
        age = axioms['age'][int(node_features[node_b][1])]

        sequence.extend([relationship, age, gender])

    # Encode the sequence
    encoded_sequence = torch.tensor([encoding.get(s, 0) for s in sequence], dtype=torch.float32)
    
    # Pad to a length of 4 (or other specified length)
    padded_sequence = F.pad(encoded_sequence, (0, max((3 * edges_away) - encoded_sequence.shape[0], 0)), 'constant', 0)

    return padded_sequence.unsqueeze(0)