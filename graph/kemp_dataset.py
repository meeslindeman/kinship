import os
import torch
import random
import numpy as np
from torch_geometric.data import Dataset
from graph.kemp_build import get_graph, update_age, update_sex 
#from uniform_build import get_graph, update_age, update_sex #for drawing

class KempGraphDataset(Dataset):
    def __init__(self, root: str, number_of_graphs: int = 5000, need_probs=None, transform=None, pre_transform=None):
        self.number_of_graphs = number_of_graphs
        self.need_probs = need_probs
        super(KempGraphDataset, self).__init__(root, transform, pre_transform)
        self.data = None
        self.process()

    @property
    def processed_file_names(self):
        return ['kinship_trees.pt']

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
    
    def generate_target(self, graph_data, node_map):
        ego_node_idx = graph_data.ego_node_idx
        num_nodes = graph_data.num_nodes

        # Convert the returned tensor to a set and exclude the ego node itself
        possible_nodes = set(range(num_nodes)) - {ego_node_idx}

        # If no nodes are found, use the fallback method to include all nodes except the ego node
        if not possible_nodes:
            possible_nodes = set(range(num_nodes)) - {ego_node_idx}

        if self.need_probs:
            # Map possible nodes to their names
            node_map_inv = {v: k for k, v in node_map.items()}
            possible_node_names = [node_map_inv[node_idx] for node_idx in possible_nodes]

            # Get the probability of each node being chosen
            probs = [self.need_probs.get(node, 0) for node in possible_node_names]

            # Normalize the probabilities
            total_prob = sum(probs)
            if total_prob > 0:
                probs = [p/ total_prob for p in probs]
            else:
                probs = [1/ len(possible_nodes) * len(possible_nodes)]
            
            target_node_name = np.random.choice(possible_node_names, p=probs)
            target_node_idx = node_map[target_node_name]
        
        else:
            # Randomly choose one of the possible target nodes
            target_node_idx = random.choice(list(possible_nodes))
        
        return target_node_idx
    
    def update_graph_attributes(self, graph_data, ego_idx, node_map):
        update_age(graph_data, ego_idx, node_map)
        update_sex(graph_data, ego_idx)

    def process(self):
        if not os.path.isfile(self.processed_paths[0]):
            self.data = []
            for i in range(self.number_of_graphs):
                graph_data, node_map = get_graph()

                ego_node_idx = 16 # Ego node index

                graph_data.ego_node_idx = ego_node_idx

                # Alternate between male and female ego nodes
                if i % 2 == 0:
                    graph_data.x[graph_data.ego_node_idx][0] = 0  # Male
                    ego_name = 'Bob'
                else:
                    graph_data.x[graph_data.ego_node_idx][0] = 1  # Female
                    ego_name = 'Alice'

                # Generate target node
                target_node_idx = self.generate_target(graph_data, node_map)
                graph_data.target_node_idx = target_node_idx
                
                # Update graph attributes based on the new ego node
                self.update_graph_attributes(graph_data, ego_node_idx, node_map)

                target = list(node_map.keys())[list(node_map.values()).index(target_node_idx)]
                graph_data.target_node = target

                graph_data.ego_node = ego_name

                self.data.append(graph_data)

            torch.save(self.data, self.processed_paths[0])
        else:
            self.data = torch.load(self.processed_paths[0])


