import os
import logging
from graph.kemp_dataset import KempGraphDataset
from analysis.save import results_to_dataframe
from archs.game import get_game
from archs.dataloader import get_loaders
from archs.train import perform_training
from typing import List
from options import Options

def run_experiment(opts: Options, target_folder: str, save: bool = True):

    logging.info(f"Running {str(opts)}")

    dataset = KempGraphDataset(root=opts.root+opts.need_probs)
    print(f"Dataset: {opts.root+opts.need_probs}")

    train_loader, valid_loader = get_loaders(opts, dataset)
    game = get_game(opts, dataset.num_node_features)
    results, trainer = perform_training(opts, train_loader, valid_loader, game)

    return results_to_dataframe(results, opts, target_folder, save=save)

def run_series_experiments(opts_list: List[Options], base_target_folder: str):

    results = []

    for opts in opts_list:
        target_folder = os.path.join(base_target_folder, opts.need_probs)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        result = run_experiment(opts, target_folder, False)
        filename = f"df_{str(opts)}.csv"
        result.to_csv(os.path.join(target_folder, filename), index=False)
        
    return results, target_folder