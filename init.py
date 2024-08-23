from typing import Dict, Optional
from graph.kemp_dataset import KempGraphDataset
from options import Options
from need_probs import get_need_probs

opts = Options()

need_probs = get_need_probs(opts.need_probs)

def process_dataset(need_probs: Optional[Dict] = None): 

    dataset = KempGraphDataset(root=opts.root+opts.need_probs, number_of_graphs=opts.number_of_graphs, need_probs=need_probs)

    total_nodes = sum(data.num_nodes for data in dataset)
    average_nodes = total_nodes / len(dataset)

    return dataset, average_nodes

dataset, average_nodes = process_dataset(need_probs=need_probs)

print(f"Dataset initialized with {opts.need_probs} need probabilities.")
print(f"Number of graphs: {len(dataset)}")
print(f"Average number of nodes: {average_nodes}")
print(f"Example graph: {dataset[0]}\n")