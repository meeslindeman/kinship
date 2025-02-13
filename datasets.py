from graph.kemp_dataset import KempGraphDataset

dataset = KempGraphDataset('data/uniform_seed1', prune=True)

print(dataset[0])