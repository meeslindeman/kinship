import os
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

def load_dataframes(folder_path: str) -> list:
    """
    Load CSV files from a given folder and return a list of dataframes and their filenames.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        tuple: A tuple containing a list of dataframes and a list of filenames.
    """
    dataframes = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            dataframes.append(df)
            filenames.append(file)
    return dataframes, filenames

def accuracy_to_bits(accuracy: float) -> float:
    """
    Convert accuracy to information gain in bits.

    Parameters:
        accuracy (float): The accuracy of the model (0 <= accuracy <= 1).
    
    Returns:
        float: The information gain in bits.
    """
    # Ensure accuracy is within bounds
    if accuracy < 0 or accuracy > 1:
        raise ValueError("Accuracy must be between 0 and 1.")
    
    # Calculate entropy for the given accuracy
    if accuracy == 0 or accuracy == 1:
        return 1  # Perfect certainty or total uncertainty
    current_entropy = -accuracy * np.log2(accuracy) - (1 - accuracy) * np.log2(1 - accuracy)
    
    # Initial entropy is 1 bit for binary classification (50% guess)
    initial_entropy = 1.0
    
    # Information gain
    information_gain = initial_entropy - current_entropy
    return information_gain


def plot_combined_scatter_final_epoch(folder_path: str, y_metric: str, save: bool = False, mode: str = 'both'):
    """
    Plot a combined scatter plot for the final epoch for a given y metric.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
        y_metric (str): Metric to plot on the y-axis.
        save (bool): Whether to save the plot as a PNG file.
        mode (str): Mode of the data ('train', 'test', or 'both').
    """
    dataframes, filenames = load_dataframes(folder_path)
    if not dataframes:
        print("No dataframes found.")
        return
    
    if mode not in ['train', 'test', 'both']:
        print("Invalid mode. Please choose 'train', 'test', or 'both'.")
        return
    
    titles_mapping = {
        'acc': 'Accuracy (bits)',
        'complexity': 'Complexity',
        'information_loss': 'Communicative Cost',
        'entropy': 'Entropy',
        'kl_accuracy': 'Accuracy (bits)'
    }

    # Create a subplot
    fig = make_subplots(rows=1, cols=1, 
                        shared_xaxes=True,
                        shared_yaxes=True,
                        horizontal_spacing=0.01, 
                        vertical_spacing=0.02,
                        x_title=titles_mapping.get('complexity'),
                        y_title=titles_mapping.get(y_metric, y_metric))

    for df, filename in zip(dataframes, filenames):
        if mode != 'both':
            df = df[df["mode"] == mode]

        max_lens = sorted(df['max_len'].unique())
        final_epoch = df['epoch'].max()

        for max_len in max_lens:
            df_max_len = df[(df['max_len'] == max_len) & (df['epoch'] == final_epoch)].dropna(subset=[y_metric])

            # Convert accuracy to bits if metric is 'acc'
            if y_metric == 'acc':
                #df_max_len[y_metric] = df_max_len[y_metric].apply(accuracy_to_bits)
                df_max_len[y_metric] = df_max_len[y_metric]


            trace = go.Scatter(x=df_max_len['complexity'], 
                               y=df_max_len[y_metric], 
                               mode='markers',
                               marker=dict(size=10, line=dict(width=1)),
                               name=f"{filename.rstrip('.csv')}")
                        
            fig.add_trace(trace, row=1, col=1)

    # Update layout
    fig.update_layout(legend=dict(
                        orientation="h",
                        xanchor="left",
                        yanchor="bottom",
                        x=0,
                        y=1),
                      height=600,
                      width=800)
    
    fig.update_xaxes(range=[0, 7])
    if y_metric == 'acc':
        fig.update_yaxes(range=[0, 4])
    else:
        fig.update_yaxes(range=[0, 4])
    
    if save:
        fig.write_image(f"plots/test.png", scale=2)
    else:
        fig.show()

# Usage
plot_combined_scatter_final_epoch('to_plot/', y_metric='kl_accuracy', save=True, mode='train')
