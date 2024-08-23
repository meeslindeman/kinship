import pandas as pd
from collections import Counter
import ast

df = pd.read_csv('results/df_maxlen=3.csv')
df = df.dropna()
df = df[df['mode'] == 'train']

# Extract the final epoch data
final_epoch_data = df.iloc[-1]

# Extract columns for the final epoch
messages = ast.literal_eval(final_epoch_data['messages'])
target_nodes = ast.literal_eval(final_epoch_data['target_node'])
ego_nodes = ast.literal_eval(final_epoch_data['ego_node'])

# Separate messages for Alice and Bob
alice_messages = []
bob_messages = []
alice_target_nodes = []
bob_target_nodes = []


for i in range(len(ego_nodes)):
    if ego_nodes[i] == 'Alice':
        alice_messages.append(tuple(messages[i]))
        alice_target_nodes.append(target_nodes[i])
    elif ego_nodes[i] == 'Bob':
        bob_messages.append(tuple(messages[i]))
        bob_target_nodes.append(target_nodes[i])

# Use Counter to match messages to target nodes
alice_counter = Counter(zip(alice_target_nodes, alice_messages))
bob_counter = Counter(zip(bob_target_nodes, bob_messages))

# Print results
print("Alice's message to target node counts:")
for k, v in alice_counter.items():
    print(f"Target Node: {k[0]}, Message: {k[1]}, Count: {v}")

print("\nBob's message to target node counts:")
for k, v in bob_counter.items():
    print(f"Target Node: {k[0]}, Message: {k[1]}, Count: {v}")

# Compare the messages for the same target nodes between Alice and Bob
print("\nDifferences in messages to the same target node:")
alice_targets = set(alice_target_nodes)
bob_targets = set(bob_target_nodes)
common_targets = alice_targets & bob_targets

for target in common_targets:
    alice_msgs = {msg for tgt, msg in alice_counter if tgt == target}
    bob_msgs = {msg for tgt, msg in bob_counter if tgt == target}
    
    if alice_msgs != bob_msgs:
        print(f"Target Node: {target}")
        print(f"  Alice's messages: {alice_msgs}")
        print(f"  Bob's messages: {bob_msgs}")