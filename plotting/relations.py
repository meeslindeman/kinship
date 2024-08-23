import matplotlib.pyplot as plt
from collections import Counter
import sys
import os

"""
Plots a distribution of relationships in the dataset.
Can be used for both, mainly useful for randomly generated graphs.
"""

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.kemp_dataset import KempGraphDataset
from analysis.relationship import get_relationship
from options import Options

opts = Options()

def plot_relationship_distribution(dataset_root: str):
    dataset = KempGraphDataset(root=dataset_root)

    # Extract all relationships using the function get_relationship and store them in a list
    relationships = [get_relationship(item) for item in dataset]
    
    # Use Counter to count the frequency of each relationship
    relationship_counts = Counter(relationships)

    sorted_relationship_counts = sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    labels, counts = zip(*sorted_relationship_counts)

    # Calculate percentages
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Relationship')
    plt.title(f'Distribution of Relationships ({len(dataset)} samples)\n{opts.need_probs}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Add percentage labels to the bars
    for bar, perc in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{perc:.2f}', ha='center', va='bottom', fontsize=8)

    plt.savefig(f'plots/relationships_{opts.need_probs}.png')
    #plt.show()

if __name__ == "__main__":
    dataset_root = opts.root+opts.need_probs  
    plot_relationship_distribution(dataset_root)