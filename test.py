from graph.kemp_dataset import KempGraphDataset

dataset = KempGraphDataset('data/uniform_seed42', prune=False, seed=42)

print(dataset[0].x)