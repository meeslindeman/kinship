import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from graph.kemp_dataset import KempGraphDataset

def draw_graph(data, nodes, save: bool = False, index: int = 0):
    plt.figure(figsize=(8, 6))

    G = nx.DiGraph()

    target_node = data.target_node_idx
    ego_node = data.ego_node_idx

    # Add nodes with age attributes
    for i, attr in enumerate(data.x):
        G.add_node(i, age=int(attr[1]))

    # Initialize dictionaries for different relationships
    lineal_labels = {}

    for start, end, attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr):
        G.add_edge(start.item(), end.item())
        if attr.tolist() == [1, 0]:
            lineal_labels[(start.item(), end.item())] = 'parent-of'

    def get_node_color(i, gender, ego_node, target_node):
        return 'tab:blue' if gender == 0 else 'tab:red'
    
    node_colors = [get_node_color(i, gender, ego_node, target_node) for i, (gender, _, _) in enumerate(data.x.tolist())]

    pos = nx.spring_layout(G, seed=42)

    # Transform the layout to plot vertically
    pos = {node: (-y, x) for node, (x, y) in pos.items()}

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if (u, v) in lineal_labels], arrows=True, arrowstyle='-|>', arrowsize=6, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if (u, v) not in lineal_labels], arrows=False, alpha=0.5)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=lineal_labels, font_size=4, label_pos=0.5)

    # Draw node labels with index and age
    relationship_labels = {0: "Y", 1: "O"}
    labels = {i: f"{nodes[i]}\n{relationship_labels[int(data.x[i, 1])]}" for i in range(len(data.x))}
    nx.draw_networkx_labels(G, pos, labels, font_size=6)

    # Create a legend for the colors
    patches = [
        mpatches.Patch(color='tab:blue', label='Male'),
        mpatches.Patch(color='tab:red', label='Female')
    ]
    plt.legend(handles=patches)

    if save:
        plt.savefig(f"graph.png", dpi=300)
        plt.close()
    else:
        plt.show()

# Usage
nodes = [
        'MM', 'MF', 'MZy', 'MBy', 'M', 'MZe', 'MBe',
        'FM', 'FF', 'FZy', 'FBy', 'F', 'FZe', 'FBe',
        'Zy', 'By', 'Bob', 'Ze', 'Be', 'ZyD', 'ZyS',
        'ByD', 'ByS', 'D', 'S', 'ZeD', 'ZeS', 'BeD', 'BeS',
        'DD', 'DS', 'SD', 'SS'
    ]

dataset = KempGraphDataset(root='data/kemp')
data = dataset[0]
draw_graph(data, nodes, save=True, index=0)