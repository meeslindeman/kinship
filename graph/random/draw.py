import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

def draw_graph(data, origin: str = 'ego', save: bool = False, index: int = 0):
    """
    Draw a graph based on the given data.

    Parameters:
        data (object): The data object containing the necessary information for drawing the graph.
        origin (str, optional): The origin node for the graph layout. Defaults to 'ego'.
        save (bool, optional): Whether to save the graph as an image. Defaults to False.
        index (int, optional): The index of the graph. Defaults to 0.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))

    G = nx.DiGraph()

    num_nodes = data.num_nodes
    target_node = data.target_node_idx
    ego_node = data.ego_node_idx

    # Add nodes with age attributes
    for i, attr in enumerate(data.x):
        G.add_node(i, age=int(attr[1]))

    # Initialize dictionaries for different relationships
    spouse_labels, lineal_labels = {}, {}

    for start, end, attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr):
        G.add_edge(start.item(), end.item())
        if attr.tolist() == [1, 0, 0]:
            spouse_labels[(start.item(), end.item())] = 'spouse'
        elif attr.tolist() == [0, 0, 1]:
            lineal_labels[(start.item(), end.item())] = 'parent/ child'

    def get_node_color(i, gender, ego_node, target_node):
        if i == ego_node:
            return 'lightblue' if gender == 0 else 'lightpink'
        elif i == target_node:
            return 'mediumblue' if gender == 0 else 'darkviolet'
        else:
            return 'tab:blue' if gender == 0 else 'tab:red'
    
    node_colors = [get_node_color(i, gender, ego_node, target_node) for i, (gender, _, _) in enumerate(data.x.tolist())]

    if origin == 'ego':
        pos = nx.bfs_layout(G, ego_node, align='vertical')
    elif origin == 'target':
        pos = nx.bfs_layout(G, target_node, align='vertical')
    else:
        pos = nx.bfs_layout(G, 0, align='vertical')

    # Transform the layout to plot vertically
    pos_vertical = {node: (y, -x) for node, (x, y) in pos.items()}

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos_vertical, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos_vertical, arrows=True, arrowstyle='-|>', arrowsize=6, alpha=0.5)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos_vertical, edge_labels=spouse_labels, font_size=6, label_pos=0.5)
    nx.draw_networkx_edge_labels(G, pos_vertical, edge_labels=lineal_labels, font_size=6, label_pos=0.5)

    # Draw node labels with index and age
    relationship_labels = {0: "Older", 1: "Equal", 2: "Younger"}
    labels = {i: f"{i}\n{relationship_labels[int(data.x[i, 1])]}" for i in range(len(data.x))}
    nx.draw_networkx_labels(G, pos_vertical, labels, font_size=6)

    # Create a legend for the colors
    patches = [
        mpatches.Patch(color='lightblue', label='Ego Male'),
        mpatches.Patch(color='lightpink', label='Ego Female'),
        mpatches.Patch(color='darkblue', label='Target Male'),
        mpatches.Patch(color='darkviolet', label='Target Female'),
        mpatches.Patch(color='tab:blue', label='Male'),
        mpatches.Patch(color='tab:red', label='Female')
    ]
    plt.legend(handles=patches)

    if save:
        plt.savefig(f"drawings/graph_{index}.png")
        plt.close()
    else:
        plt.show()