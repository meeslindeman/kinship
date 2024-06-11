import random
import torch

def select_distractors(data, n_distractors):
    node_list = list(range(data.num_nodes))
    adjusted_target_node_idx = data.target_node_idx + data.ptr[:-1]
    target_list = adjusted_target_node_idx.tolist()

    # Remove target nodes from the node list
    filtered_node_list = [node for node in node_list if node not in target_list]

    ptr_values = data.ptr.tolist()[:-1]

    # Initialize the output list for distractors and target nodes
    distract_and_targets = []

    for idx, (graph_start, graph_end) in enumerate(zip(ptr_values, ptr_values[1:] + [data.num_nodes])):
        # Segment's nodes excluding targets
        graph_nodes = [node for node in range(graph_start, graph_end) if node not in target_list]
        
        # Ensure we select up to n_distractors, adjusting for graphs with fewer eligible nodes
        num_distractors = min(n_distractors, len(graph_nodes))
        graph_segment = []
        if num_distractors > 0:
            dist = random.sample(graph_nodes, num_distractors)
            graph_segment.extend(dist)

        # Insert the target node at the beginning of the graph segment
        graph_segment.insert(0, target_list[idx])
        distract_and_targets.extend(graph_segment)

    return torch.tensor(distract_and_targets), torch.tensor(target_list)