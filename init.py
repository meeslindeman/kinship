import os
from graph.kemp_dataset import KempGraphDataset
from options import Options
from need_probs import get_need_probs

opts = Options()

need_probs = get_need_probs(opts.need_probs)

def initialize_dataset_if_needed(opts: Options):
    """Check if dataset exists; initialize it if not."""
    dataset_path = os.path.join(opts.root, f"{opts.need_probs}_seed{opts.data_seed}")
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Initializing dataset...")
        need_probs = get_need_probs(opts.need_probs)
        dataset = KempGraphDataset(
            root=dataset_path,
            number_of_graphs=opts.number_of_graphs,
            need_probs=need_probs,
            seed=opts.data_seed
        )
        print(f"Dataset {need_probs} initialized at {dataset_path}.")
        print(f"Number of graphs: {len(dataset)}")
    else:
        print(f"Dataset found at {dataset_path}. Skipping initialization.")
