import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import sys
import os

"""
Plots a distribution of target nodes in the dataset.
Mainly used for kemp styled kinship tree.
"""

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.kemp_dataset import KempGraphDataset
from options import Options

opts = Options()

def plot_target_node_distribution(dataset_root: str):
    # Initialize the dataset
    dataset = KempGraphDataset(root=dataset_root)

    # Get the length of the dataset
    dataset_length = len(dataset)

    # Collect target nodes
    target_nodes = [dataset[i].target_node for i in range(dataset_length)]

    # Count the frequency of each target node
    target_node_counts = Counter(target_nodes)

    # Convert the counts to a DataFrame for easy plotting
    df_counts = pd.DataFrame(target_node_counts.items(), columns=['Target Node', 'Frequency'])

    # Calculate percentages
    df_counts['Percentage'] = (df_counts['Frequency'] / dataset_length) * 100

    # Sort the DataFrame by frequency in descending order
    df_counts = df_counts.sort_values(by='Frequency', ascending=False)

    # Plot the distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_counts['Target Node'], df_counts['Frequency'], color='skyblue')
    plt.xlabel('Target Node')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of target nodes ({len(dataset)} samples)\n{opts.need_probs}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Add percentages on top of each bar
    for bar, perc in zip(bars, df_counts['Percentage']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{perc:.2f}', ha='center', va='bottom', fontsize=8)

    plt.savefig(f'plots/targets_{opts.need_probs}.png')
    #plt.show()

if __name__ == "__main__":
    dataset_root = opts.root+opts.need_probs  
    plot_target_node_distribution(dataset_root)
