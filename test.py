from graph.kemp_dataset import KempGraphDataset

dataset = KempGraphDataset('data/uniform_seed42', prune=False, seed=42)

plot_interval = len(dataset) // 50 // 2

print(plot_interval)