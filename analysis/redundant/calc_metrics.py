import pandas as pd
import numpy as np
import ast
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

df = pd.read_csv('results/single_dataframe_gs_transformer.csv')
df = df[df['mode'] == 'test']
df = df.iloc[-1]

messages = ast.literal_eval(df['messages'])
targets = ast.literal_eval(df['target_node'])

# Convert lists in 'messages' to tuples
messages = [tuple(msg) for msg in messages]

df = pd.DataFrame({'messages': messages, 'target_node': targets})

# Method: Kemp et al. - Complexity as the number of distinct terms
complexity_kemp = df['messages'].nunique()

# Method: IB - Complexity as the entropy of the message distribution
message_counts = df['messages'].value_counts(normalize=True).values
complexity_ib = entropy(message_counts)

# Calculate the probability distribution of the target nodes
target_node_counts = df['target_node'].value_counts(normalize=True).values

# Function to calculate Kullback-Leibler divergence with epsilon to handle zero values
def kl_divergence(p, q, epsilon=1e-10):
    p = np.asarray(p, dtype=np.float32) + epsilon
    q = np.asarray(q, dtype=np.float32) + epsilon
    return np.sum(p * np.log(p / q))

# Method: Kemp et al. - Informativeness as expected communicative cost
# Group by target_node and message to get joint probabilities
joint_prob = df.groupby(['target_node', 'messages']).size().unstack(fill_value=0)
joint_prob = joint_prob / joint_prob.sum().sum()

# Get the probability distribution of messages given target nodes
conditional_prob = joint_prob.div(joint_prob.sum(axis=1), axis=0).fillna(0).values

# Calculate the communicative costs
communicative_costs = []
for i in range(len(target_node_counts)):
    communicative_costs.append(kl_divergence(conditional_prob[i], message_counts))

informativeness_kemp = np.sum(target_node_counts * communicative_costs)

# Method: IB - Informativeness as mutual information
# Convert messages and target_node to categorical codes for mutual_info_score
messages_coded = df['messages'].astype('category').cat.codes
target_nodes_coded = df['target_node'].astype('category').cat.codes

informativeness_ib = mutual_info_score(target_nodes_coded, messages_coded)

# Output the results
print(f"Complexity (Kemp et al.): {complexity_kemp}")
print(f"Complexity (IB): {complexity_ib}")
print(f"Informativeness (Kemp et al.): {informativeness_kemp}")
print(f"Informativeness (IB): {informativeness_ib}")
