from graph.kemp_dataset import KempGraphDataset

dataset = KempGraphDataset('data/uniform_seed0', prune=False, seed=0)
dataset1 = KempGraphDataset('data/uniform_seed25', prune=False, seed=25)
dataset2 = KempGraphDataset('data/uniform_seed42', prune=False, seed=42)
dataset3 = KempGraphDataset('data/uniform_seed100', prune=False, seed=100)

from collections import Counter

# Assuming `dataset` is a list of dictionaries or objects with a 'target_node' field
target_node_counter = Counter()

# Loop through the dataset and count occurrences of each target_node
for item in dataset:
    target_node_counter[item['target_node']] += 1

# Print the distribution of target_node values
for target_node, count in target_node_counter.items():
    print(f"Target Node: {target_node}, Count: {count}")
