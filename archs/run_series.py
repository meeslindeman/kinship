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
        'batch_size': int(opts.batch_size),
        'random_seed': int(opts.random_seed),
        'with_vq': bool(opts.with_vq),
        'codebook_size': int(opts.codebook_size)
        }

        # init wandb
        wb.init(project="referential_game",
                name=f"{opts.need_probs}_vq={opts.with_vq}_{opts.mode}",
                config=params,
                settings=wb.Settings(_disable_stats=True) # disable system metrics
            )

    logging.info(f"Running {str(opts)}")

    dataset = KempGraphDataset(root=opts.root+opts.need_probs, prune=opts.prune_graph)
    print(f"Dataset: {opts.root+opts.need_probs}")

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
            
            # Add data for this epoch to the scatter plot table
            table.add_data(complexity, info_loss, epoch)

            wb.log({
                "eval metrics/Epoch": epoch, 
                "eval metrics/Complexity": complexity,
                "eval metrics/Information Loss": info_loss
            })

        wb.log({"eval metrics/": table})

        wb.finish()

    return metrics_df

def run_series_experiments(opts_list: List[Options], base_target_folder: str):

    results = []

    for opts in opts_list:
        target_folder = os.path.join(base_target_folder, opts.need_probs)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        metrics = run_experiment(opts, target_folder, False)
        filename = f"df_{str(opts)}.csv"
        metrics.to_csv(os.path.join(target_folder, filename), index=False)

    return results, target_folder