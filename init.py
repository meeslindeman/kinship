from typing import Dict, Optional
from graph.kemp_dataset import KempGraphDataset
from options import Options
from need_probs import get_need_probs

opts = Options()

need_probs = get_need_probs(opts.need_probs)

import os
from typing import Dict, Optional
from options import Options
from graph.kemp_dataset import KempGraphDataset
from need_probs import get_need_probs

def initialize_dataset_if_needed(opts: Options):
    """Check if dataset exists; initialize it if not."""
    dataset_path = os.path.join(opts.root, opts.need_probs)
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Initializing dataset...")
        
        need_probs = get_need_probs(opts.need_probs)
        dataset, average_nodes = process_dataset(opts, need_probs)

        print(f"Dataset initialized with {opts.need_probs} need probabilities.")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Average number of nodes: {average_nodes}")
        print(f"Example graph: {dataset[0]}\n")
    else:
        print(f"Dataset found at {dataset_path}. Skipping initialization.")

def process_dataset(opts: Options, need_probs: Optional[Dict] = None):
    """Initialize the dataset and calculate statistics."""
    dataset = KempGraphDataset(
        root=os.path.join(opts.root, opts.need_probs),
        number_of_graphs=opts.number_of_graphs,
        need_probs=need_probs
    )

    total_nodes = sum(data.num_nodes for data in dataset)
    average_nodes = total_nodes / len(dataset)

    return dataset, average_nodes