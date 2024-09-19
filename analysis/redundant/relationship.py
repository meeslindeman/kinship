from torch_geometric.utils import k_hop_subgraph, to_networkx
import networkx as nx

def gather_relationship_data(graph, path, edge_index, edge_attr):
    """Gather the relationships and attributes between nodes along the path and return them as dictionaries."""

    # The mappings provided in the user's example
    gender_map = {0: 'm', 1: 'f'}
    # age_map = {0: 'older', 1: 'equal', 2: 'younger'}
    # relationship_map = {
    #     (1., 0., 0.): 'married',
    #     (0., 1., 0.): 'child-of',
    #     (0., 0., 1.): 'parent-of'}

    age_map = {0: 'younger', 1: 'older'}
    relationship_map = {
        (1., 0.): 'parent-of',
        (0., 1.): 'child-of'}
    
    # Initialize dictionaries for node information and relationships
    node_info = {}
    relationships = {}

    # Iterate over consecutive pairs of nodes in the path
    for i in range(len(path) - 1):
        node_a = path[i]
        node_b = path[i + 1]

        # Find the relationship between node_a and node_b using the edge_index and edge_attr
        for j in range(edge_index.shape[1]):
            if edge_index[0, j] == node_a and edge_index[1, j] == node_b:
                relationship = relationship_map.get(tuple(edge_attr[j].tolist()), "unknown")
                break
        else:
            relationship = "unknown"

        # Store the relationship from node_a to node_b
        relationships[(node_a, node_b)] = relationship

        # Store node_a's gender and age
        if node_a not in node_info:
            gender_a = gender_map[int(graph['x'][node_a][0])]
            age_a = age_map[int(graph['x'][node_a][1])]
            node_info[node_a] = {'gender': gender_a, 'age': age_a}

    # Add the target node's attributes
    node_b = path[-1]
    if node_b not in node_info:
        gender_b = gender_map[int(graph['x'][node_b][0])]
        age_b = age_map[int(graph['x'][node_b][1])]
        node_info[node_b] = {'gender': gender_b, 'age': age_b}

    return node_info, relationships

def get_relationship_name(path, relationships, node_info):
    degree = len(path) - 1

    if degree == 1:
        # Direct relationships (parents/children, spouse)
        rel_type = relationships.get((path[0], path[1]), 'unknown')
        if rel_type == 'married':
            return "spouse"
        elif rel_type == 'child-of':
            return "son" if node_info[path[0]]['gender'] == 'm' else "daughter"
        elif rel_type == 'parent-of':
            return "father" if node_info[path[0]]['gender'] == 'm' else "mother"

    elif degree == 2:
        rel_1 = relationships.get((path[0], path[1]), 'unknown')
        rel_2 = relationships.get((path[1], path[2]), 'unknown')
        
        # Grandparent and grandchild relationships
        if rel_1 == 'parent-of' and rel_2 == 'parent-of':
            if node_info[path[1]]['gender'] == 'm':
                return "paternalgrandfather" if node_info[path[0]]['gender'] == 'm' else "paternalgrandmother"
            elif node_info[path[1]]['gender'] == 'f':
                return "maternalgrandfather" if node_info[path[0]]['gender'] == 'm' else "maternalgrandmother"
        elif rel_1 == 'child-of' and rel_2 == 'child-of':
            return "grandson" if node_info[path[0]]['gender'] == 'm' else "granddaughter"
        
        # Sibling relationship
        if rel_1 == 'child-of' and rel_2 == 'parent-of':
            if node_info[path[0]]['age'] == 'older':
                return "olderbrother" if node_info[path[0]]['gender'] == 'm' else "oldersister"
            elif node_info[path[0]]['age'] == 'younger':
                return "youngerbrother" if node_info[path[0]]['gender'] == 'm' else "youngersister"
            else:
                return "equalbrother" if node_info[path[0]]['gender'] == 'm' else "equalsister"
            
        # In-law relationship
        if rel_1 == 'married' and rel_2 == 'child-of':
            return "father-in-law" if node_info[path[0]]['gender'] == 'm' else "mother-in-law"
        elif rel_1 == 'parent-of' and rel_2 == 'married':
            return "son-in-law" if node_info[path[0]]['gender'] == 'm' else "daughter-in-law"

    elif degree == 3:
        rel_1 = relationships.get((path[0], path[1]), 'unknown')
        rel_2 = relationships.get((path[1], path[2]), 'unknown')
        rel_3 = relationships.get((path[2], path[3]), 'unknown')

        # Niece/nephew relationship
        if rel_1 == 'child-of' and rel_2 == 'child-of' and rel_3 == 'parent-of':
            return "nephew" if node_info[path[0]]['gender'] == 'm' else "niece"
        
        # Uncle/aunt relationship
        if rel_1 == 'child-of' and rel_2 == 'parent-of' and rel_3 == 'parent-of':
            if node_info[path[2]]['gender'] == 'm':
                return "paternaluncle" if node_info[path[0]]['gender'] == 'm' else "paternalaunt"
            elif node_info[path[2]]['gender'] == 'f':
                return "maternaluncle" if node_info[path[0]]['gender'] == 'm' else "maternalaunt"
        
        # In-law relationship
        if rel_1 == 'married' and rel_2 == 'child-of' and rel_3 == 'parent-of':
            return "spouses-brother" if node_info[path[0]]['gender'] == 'm' else "spouses-sister"
        elif rel_1 == 'child-of' and rel_2 == 'parent-of' and rel_3 == 'married':
            return "brothers-wife" if node_info[path[0]]['gender'] == 'm' else "sisters-husband"
    
    print("Unknown relationship path: ", path, relationships, node_info)

    return "unknown"


def get_relationship(graph):
    """Determine the familial relationship between ego and target nodes."""
    graphx = to_networkx(graph)
    paths = list(nx.all_shortest_paths(graphx, source=graph.target_node_idx, target=graph.ego_node_idx))

    # Find the path of relationships between ego and target
    path = paths[0] 

    # Gather the relationship data from the graph
    node_info, relationships = gather_relationship_data(graph, path, graph.edge_index, graph.edge_attr)

    # Determine the relationship name based on the path and relationship data
    relationship = get_relationship_name(path, relationships, node_info)

    return relationship