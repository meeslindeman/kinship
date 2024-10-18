from typing import Any, List, Optional, Sequence, Union
from sklearn.model_selection import train_test_split
from options import Options

import torch.utils.data
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.on_disk_dataset import OnDiskDataset

class Collater:
    def __init__(
        self,
        game_size: int,  # the number of graphs for a game
        batch_size: int,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.game_size = game_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            batch = batch[:((len(batch) // self.game_size) * self.game_size)]  # we throw away the last batch_size % game_size
            batch = Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
            # Returns a tuple (sender_input, labels, receiver_input, aux_input), aux_input is used to store minibatch of graphs
            return (
                torch.zeros(self.batch_size, dtype=torch.long), #batch.sequence, # sender input -> sequence of the graph
                torch.zeros(self.batch_size, dtype=torch.long), # Needs to be zeros times batch size!
                None,  
                batch  
            )

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        if isinstance(self.dataset, OnDiskDataset):
            return self(self.dataset.multi_get(batch))
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        game_size: int,  # the number of graphs for a game
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.game_size = game_size
        self.batch_size = batch_size
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.collator = Collater(game_size, batch_size, dataset, follow_batch, exclude_keys)

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn,
            **kwargs,
        )

def create_eval_data(dataset):
    nodes = set([
        'MM', 'MF', 'MZy', 'MBy', 'M', 'MZe', 'MBe',
        'FM', 'FF', 'FZy', 'FBy', 'F', 'FZe', 'FBe',
        'Zy', 'By', 'Ego', 'Ze', 'Be', 'ZyD', 'ZyS',
        'ByD', 'ByS', 'D', 'S', 'ZeD', 'ZeS', 'BeD', 'BeS',
        'DD', 'DS', 'SD', 'SS'
    ])

    filtered_data = []
    
    for data in dataset:
        # Convert target_node to a scalar if it's a tensor
        if isinstance(data.target_node, torch.Tensor):
            target_node = data.target_node.item()  # Convert tensor to scalar
        else:
            target_node = data.target_node  # Use target_node as-is if it's a string or integer

        # If the target_node is in the set and hasn't been found yet, add it to the filtered_data
        if target_node in nodes:
            filtered_data.append(data)
            nodes.remove(target_node)  # Remove the node from the set to ensure uniqueness

        # Stop once we've found all 32 targets
        if len(filtered_data) == 32:
            break
    return filtered_data

def get_loaders(opts, dataset):
    _, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(game_size=1, dataset=dataset, batch_size=opts.batch_size, shuffle=True)
    val_loader = DataLoader(game_size=1, dataset=val_data, batch_size=opts.batch_size, shuffle=True)

    eval_data = create_eval_data(dataset)
    eval_loader = DataLoader(game_size=1, dataset=eval_data, batch_size=opts.eval_batch_size, shuffle=False)

    return train_loader, val_loader, eval_loader