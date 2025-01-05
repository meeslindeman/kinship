import json
import pandas as pd
from options import Options
import os

def results_to_dataframes(results: list, opts: Options, target_folder: str, save: bool = True):
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Extract parameters
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
        'data_seed': int(opts.data_seed)
    }

    # Initialize lists for dataframes
    metrics = []
    evaluation = []
    counts = []

    for result in results:
        if 'message_counts' in result:
            epoch = result['epoch']
            message_counts = result['message_counts']

            for target, messages in message_counts.items():
                for message, count in messages.items():
                    counts.append({
                        'epoch': epoch,
                        'target': target,
                        'message': message,
                        'count': count
                    })

        if 'evaluation' in result:
            for eval_item in result['evaluation']:
                evaluation.append({
                    'Epoch': eval_item.get('epoch'),
                    'Ego Node': eval_item.get('ego_node'),
                    'Target Node': eval_item.get('target_node'),
                    'Message': eval_item.get('message'),  # Could be a list or array
                    'Receiver Output': eval_item.get('receiver_output'),  # Long list of outputs
                    'Predicted Label': eval_item.get('predicted_label'),
                    'Correct': eval_item.get('correct'),
                    'Complexity': result.get('complexity'),
                    'Information Loss': result.get('information_loss')
                })

        metrics.append({
            'epoch': result['epoch'],
            'mode': result['mode'],
            'loss': result['loss'],
            **{k: v for k, v in result.items() if k not in ['epoch', 'mode', 'loss', 'message_counts', 'evaluation']}
        })

    metrics_df = pd.DataFrame(metrics)
    evaluation_df = pd.DataFrame(evaluation)
    counts_df = pd.DataFrame(counts)

    for key, value in params.items():
        metrics_df[key] = value

    # Save DataFrames
    if save:
        metrics_df.to_csv(f'{target_folder}/metrics_{opts.mode}.csv', index=False)
        counts_df.to_csv(f'{target_folder}/counts.csv', index=False)
        evaluation_df.to_csv(f'{target_folder}/evaluation.csv', index=False)

    return metrics_df, counts_df, evaluation_df
