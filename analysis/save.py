import json
import pandas as pd
from options import Options

def results_to_dataframe(results: list[dict], opts: Options, target_folder: str, save: bool = True) -> pd.DataFrame:
    """
    Converts a list of results into a pandas DataFrame and adds additional columns from options.
    
    Args:
        results (list[dict]): A list of dictionaries representing the results.
        n_nodes: The number of nodes.
        opts (Options): An instance of the Options class.
        target_folder (str): The target folder to save the DataFrame.
        save (bool, optional): Whether to save the DataFrame to a CSV file. Defaults to True.
    
    Returns:
        pd.DataFrame: The resulting DataFrame.
    """

    for result in results:
        if 'messages' in result:
            result['messages'] = list(result['messages'])
        if 'targets' in result:
            result['targets'] = list(result['targets'])
        if 'ego_nodes' in result:
            result['ego_nodes'] = list(result['ego_nodes'])
        if 'complexity' in result:
            result['complexity'] = float(result['complexity'])
        if 'entropy' in result:
            result['entropy'] = float(result['entropy'])
        if 'information_loss' in result:
            result['information_loss'] = float(result['information_loss'])
        if 'IB_bottleneck' in result:
            result['IB_bottleneck'] = float(result['IB_bottleneck'])
            
    # Create initial DataFrame
    initial = pd.DataFrame({'mode': ['train', 'test'], 'epoch': [0, 0], 'acc': [0, 0]})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = pd.concat([initial, results_df], ignore_index=True)

    # Add additional columns from opts
    results_df['distractors'] = int(opts.distractors)
    results_df['vocab_size'] = int(opts.vocab_size)
    results_df['hidden_size'] = int(opts.hidden_size)
    results_df['n_epochs'] = int(opts.n_epochs)
    results_df['embedding_size'] = int(opts.embedding_size)
    results_df['heads'] = int(opts.heads)
    results_df['max_len'] = int(opts.max_len)
    results_df['sender_cell'] = str(opts.sender_cell)
    results_df['train_method'] = str(opts.mode)
    results_df['batch_size'] = int(opts.batch_size)
    results_df['random_seed'] = int(opts.random_seed)

    if save:
        results_df.to_csv(f'{target_folder}/df_{opts}.csv', index=False)

    return results_df