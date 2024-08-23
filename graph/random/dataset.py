import os
import torch
import random
from torch_geometric.data import Dataset
from torch_geometric.utils import k_hop_subgraph
from graph.build import create_family_tree, create_data_object
from graph.sequence import generate_sequence
from graph.sequence_old import process_graph_to_sequence
# from tree import get_relationship

class FamilyGraphDataset(Dataset):
    """
    Dataset class for generating family graph data.

    Args:
        root (str): Root directory path.
        number_of_graphs (int): Number of graphs to generate.
        generations (int): Number of generations in each family tree.

    Returns:
        Data object containing the family graph data.
    """
    def __init__(self, root: str, number_of_graphs: int = 3200, generations: int = 1, padding_len: int = 80, edges_away: int = 2, transform=None, pre_transform=None):
        self.number_of_graphs = number_of_graphs
        self.generations = generations
        self.padding_len = padding_len
        self.edges_away = edges_away
        super(FamilyGraphDataset, self).__init__(root, transform, pre_transform)
        self.data = None
        self.process()

    @property
    def processed_file_names(self):
        return ['family_graphs.pt']

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
    
    def process_age_relations(self, graph_data):
        ego_age = graph_data.x[graph_data.ego_node_idx, 1].item()
        age_relations = []
        for age in graph_data.x[:, 1]:
            if age < ego_age:
                age_relations.append(2)  # Younger
            elif age > ego_age:
                age_relations.append(0)  # Older
            else:
                age_relations.append(1)  # Equal
        graph_data.x[:, 1] = torch.tensor(age_relations, dtype=graph_data.x.dtype, device=graph_data.x.device)
            
    def process(self):
        if not os.path.isfile(self.processed_paths[0]):
            self.data = []
            for _ in range(self.number_of_graphs):
                family_tree = create_family_tree(self.generations)
                graph_data = create_data_object(family_tree)

                # Generate ego node
                ego_node_idx = self.generate_ego(graph_data.num_nodes)
                graph_data.ego_node_idx = ego_node_idx

                # Generate target node
                target_node_idx = self.generate_target(graph_data, self.edges_away)
                graph_data.target_node_idx = target_node_idx

                # Process age relations
                self.process_age_relations(graph_data)

                # Process graph to sequence
                sequence = generate_sequence(graph_data, self.edges_away)
                graph_data.sequence = sequence

                # Process relationship
                #relationship = get_relationship(graph_data)
                #graph_data.relationship = relationship

                self.data.append(graph_data)

            torch.save(self.data, self.processed_paths[0])
        else:
            self.data = torch.load(self.processed_paths[0])