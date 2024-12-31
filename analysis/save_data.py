import json
import os

def save_metadata(dataset_path, seed, number_of_graphs):
    metadata = {
        "seed": seed,
        "number_of_graphs": number_of_graphs
    }
    metadata_path = os.path.join(dataset_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
