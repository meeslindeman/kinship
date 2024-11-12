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
                name=f"kinship_{opts.need_probs}_vq={opts.with_vq}",
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

        # message_tracker = defaultdict(lambda: defaultdict(list))

        # for epoch in counts_df['epoch'].unique():
        #     epoch_counts = counts_df[counts_df['epoch'] == epoch]
            
        #     for target in epoch_counts['target'].unique():
        #         target_data = epoch_counts[epoch_counts['target'] == target]
                
        #         # Track each message count for this target across epochs
        #         for _, row in target_data.iterrows():
        #             message = int(row['message'])  # Ensure message is an integer
        #             count = row['count']
        #             message_tracker[target][message].append((epoch, count))
        
        # for target, messages in message_tracker.items():
        #     # Prepare data for plotting
        #     log_data = {"epoch": []}
        #     for message, counts in messages.items():
        #         epochs, message_counts = zip(*counts)
        #         log_data["epoch"] = list(epochs)
        #         log_data[f"message_{message}"] = list(message_counts)
            
        #     # Log a single line plot per target
        #     wb.log({f"messages/target_{target}": wb.plot.line_series(
        #         xs=log_data["epoch"],
        #         ys=[log_data[f"message_{m}"] for m in messages.keys()],
        #         keys=[f"message_{m}" for m in messages.keys()],
        #         title=f"Message Counts for Target {target}",
        #         xname="Epochs"
        #     )})
            
        for epoch in evaluation_df['Epoch'].unique():
            epoch_data = evaluation_df[evaluation_df['Epoch'] == epoch]
            table = wb.Table(columns=["Epoch", "Target Node", "Message"])
            for _, row in epoch_data.iterrows():
                table.add_data(row['Epoch'], row['Target Node'], row['Message'][0])
            
            wb.log({f"eval_messages/epoch_{epoch}": table})

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