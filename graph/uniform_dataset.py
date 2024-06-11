import os
import torch
import random
from torch_geometric.data import Dataset
from torch_geometric.utils import k_hop_subgraph
from graph.uniform_build import get_graph, update_age, update_sex 

class UniformFamilyGraphDataset(Dataset):
    """
    Dataset class for generating a uniform family graph data.

    Args:
        root (str): Root directory path.
        number_of_graphs (int): Number of graphs to generate.
        padding_len (int): Padding length for sequences.
        edges_away (int): Number of hops for k-hop subgraph.

    Returns:
        Data object containing the family graph data.
    """
    def __init__(self, root: str, number_of_graphs: int = 3200, edges_away: int = 2, transform=None, pre_transform=None):
        self.number_of_graphs = number_of_graphs
        self.edges_away = edges_away
        super(UniformFamilyGraphDataset, self).__init__(root, transform, pre_transform)
        self.data = None
        self.process()

    @property
    def processed_file_names(self):
        return ['uniform_family_graphs.pt']

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
    
    def generate_ego(self, num_nodes):
        ego_node_idx = random.randint(0, num_nodes - 1)
        return ego_node_idx

    def generate_target(self, graph_data, edges_away: int = 2):
        ego_node_idx = graph_data.ego_node_idx
        edge_index = graph_data.edge_index
        num_nodes = graph_data.num_nodes

        # Extract the k-hop subgraph using the torch_geometric utility
        subset, _, _, _ = k_hop_subgraph(ego_node_idx, num_hops=edges_away, edge_index=edge_index, relabel_nodes=False)

        # Convert the returned tensor to a set and exclude the ego node itself
        possible_nodes = set(subset.tolist()) - {ego_node_idx}

        # If no nodes are found, use the fallback method to include all nodes except the ego node
        if not possible_nodes:
            possible_nodes = set(range(num_nodes)) - {ego_node_idx}

        # Randomly choose one of the possible target nodes
        target_node_idx = random.choice(list(possible_nodes))
        return target_node_idx
    
    def update_graph_attributes(self, graph_data, ego_node, node_map):
        update_age(graph_data, ego_node, node_map)
        update_sex(graph_data, ego_node, node_map)

    def process(self):
        if not os.path.isfile(self.processed_paths[0]):
            self.data = []
            for _ in range(self.number_of_graphs):
                graph_data, node_map = get_graph()

                # Generate ego node
                ego_node_idx = self.generate_ego(graph_data.num_nodes)
                # graph_data.ego_node_idx = ego_node_idx

                # Ego is Bob
                graph_data.ego_node_idx = 16

                # Generate target node
                target_node_idx = self.generate_target(graph_data, self.edges_away)
                graph_data.target_node_idx = target_node_idx
                
                # Update graph attributes based on the new ego node
                self.update_graph_attributes(graph_data, 'Bob', node_map)

                # ego = list(node_map.keys())[list(node_map.values()).index(ego_node_idx)]
                # graph_data.ego_node = ego

                target = list(node_map.keys())[list(node_map.values()).index(target_node_idx)]
                graph_data.target_node = target

                self.data.append(graph_data)

            torch.save(self.data, self.processed_paths[0])
        else:
            self.data = torch.load(self.processed_paths[0])


