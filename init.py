from graph.dataset import FamilyGraphDataset
from graph.uniform_dataset import UniformFamilyGraphDataset
from options import Options

"""
This script is used to initialise the dataset and print some statistics about it.
"""

opts = Options()

def process_dataset():
    if opts.root == 'data/':
        dataset = FamilyGraphDataset(root=opts.root, number_of_graphs=opts.number_of_graphs, generations=opts.generations, edges_away=opts.edges_away)
    elif opts.root == 'data_uniform/':
        dataset = UniformFamilyGraphDataset(root=opts.root, number_of_graphs=opts.number_of_graphs, edges_away=opts.edges_away)
    else:
        raise ValueError(f"Invalid root: {opts.root}")

    total_nodes = sum(data.num_nodes for data in dataset)
    average_nodes = total_nodes / len(dataset)

    return dataset, average_nodes

dataset, average_nodes = process_dataset()

print(f"Number of graphs: {len(dataset)}")
print(f"Average number of nodes: {average_nodes}")
print(f"Example graph: {dataset[0]}")