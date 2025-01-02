import os
import logging
import wandb as wb
from collections import defaultdict

from graph.kemp_dataset import KempGraphDataset
from analysis.save import results_to_dataframes
from archs.game import get_game
from archs.dataloader import get_loaders
from archs.train import perform_training
from typing import List
from options import Options

# WANDB FUNCTION FOR READABILITY

def run_experiment(opts: Options, target_folder: str, save: bool = True):
    if opts.log_wandb:
        params = {
        'distractors': int(opts.distractors),
        'vocab_size': int(opts.vocab_size),
        'hidden_size': int(opts.hidden_size),
        'n_epochs': int(opts.n_epochs),
        'embedding_size': int(opts.embedding_size),
        'heads': int(opts.heads),
        'max_len': int(opts.max_len),
        'sender_cell': str(opts.sender_cell),
        'train_method': str(opts.mode),
        'layer': str(opts.layer),
        'random_seed': int(opts.random_seed),
        'data_seed': int(opts.data_seed),
        }

        # init wandb
        wb.init(project="kinship",
                name=f"{opts.mode}_{opts.layer}_dist={opts.distractors}",
                config=params,
                settings=wb.Settings(_disable_stats=True) # disable system metrics
            )

    logging.info(f"Running {str(opts)}")

    dataset_path = os.path.join(opts.root, f"{opts.need_probs}_seed{opts.data_seed}")
    dataset = KempGraphDataset(dataset_path, prune=opts.prune_graph, seed=opts.data_seed)
    print(f"Dataset: {dataset_path}")

    train_loader, valid_loader, eval_loader = get_loaders(opts, dataset)
    game = get_game(opts, dataset.num_node_features)
    results, trainer = perform_training(opts, train_loader, valid_loader, eval_loader, game)

    metrics_df, counts_df, evaluation_df = results_to_dataframes(results, opts, target_folder, save)

    if opts.log_wandb:
        for _, row in metrics_df.iterrows():
            if row['mode'] == 'train':
                log_data = {
                    'epoch': row['epoch'],
                    'metrics/train/loss': row['loss'],
                    'metrics/train/accuracy': row['acc'],
                    'metrics/evaluation/eval_acc': row.get('eval_acc', None)
                }
            elif row['mode'] == 'test':
                log_data = {
                    'epoch': row['epoch'],
                    'metrics/test/loss': row['loss'],
                    'metrics/test/accuracy': row['acc']
                }

            wb.log(log_data)

        table = wb.Table(columns=[ "Complexity", "Information Loss", "Epoch"])

        # Iterate over all epochs and accumulate data
        for epoch in evaluation_df['Epoch'].unique():
            # Filter data for the current epoch
            epoch_data = evaluation_df[evaluation_df['Epoch'] == epoch]
            
            # Extract unique Complexity and Information Loss for the epoch
            complexity = epoch_data['Complexity'].iloc[0]
            info_loss = epoch_data['Information Loss'].iloc[0]

            wb.log({
                "eval metrics/Epoch": epoch, 
                "eval metrics/Complexity": complexity,
                "eval metrics/Information Loss": info_loss
            })

            table.add_data(complexity, info_loss, epoch)

        #wb.log({"eval metrics/": table})
        wb.log({"eval metrics/": wb.plot.scatter(table, x="Complexity", y="Information Loss", title="Complexity vs Information Loss")})

        wb.finish()

    return metrics_df

def run_series_experiments(opts_list: List[Options], base_target_folder: str):

    results = []

    for opts in opts_list:
        target_folder = os.path.join(base_target_folder, f"{opts.need_probs}_seed_{opts.data_seed}")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        metrics = run_experiment(opts, target_folder, False)
        filename = f"df_{str(opts)}.csv"
        metrics.to_csv(os.path.join(target_folder, filename), index=False)

    return results, target_folder