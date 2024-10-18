import random
import torch

def select_distractors(data, n_distractors, evaluation=False):
    adjusted_target_node_idx = data.target_node_idx + data.ptr[:-1]
    adjusted_ego_node_idx = data.ego_node_idx + data.ptr[:-1]
    target_list = adjusted_target_node_idx.tolist()
    ego_list = adjusted_ego_node_idx.tolist()

    ptr_values = data.ptr.tolist()[:-1]

    # Initialize the output list for distractors and target nodes
    distract_and_targets = []

    for idx, (graph_start, graph_end) in enumerate(zip(ptr_values, ptr_values[1:] + [data.num_nodes])):
        target_node = target_list[idx]
        ego_node = ego_list[idx]

        graph_nodes = [node for node in range(graph_start, graph_end) if node != target_node and node != ego_node]
        
        if evaluation:
            random.shuffle(graph_nodes)
            dist = graph_nodes
        else:
            # Ensure we select up to n_distractors, adjusting for graphs with fewer eligible nodes
            num_distractors = min(n_distractors, len(graph_nodes))
            dist = random.sample(graph_nodes, num_distractors) if num_distractors > 0 else []

        # Insert the target node at the beginning of the graph segment
        graph_segment = [target_list[idx]] + dist
        distract_and_targets.extend(graph_segment)

    return torch.tensor(distract_and_targets), torch.tensor(target_list)