"""
Old logger functions that were used to calculate info loss and complexity trough summation of games.
Use for reference.
"""

import math
import egg.core as core
from collections import Counter
from torch.distributions import Categorical
import torch

def _calculate_complexity(self, logs: core.Interaction):
    targets = [target_node for batch in logs.aux_input for target_node in batch['target_node']]
    
    if self.options.mode == 'gs': 
        messages = (logs.message.argmax(dim=-1)).tolist() 
    elif self.options.mode == 'rf': 
        # Messages already outputed with symbols
        messages = logs.message.tolist()

    # Convert messages to tuples for hashing
    messages = [tuple(m) for m in messages]

    # Calculate p(m) - frequency of each target
    target_freq = Counter(targets)
    p_m = {target: count / len(targets) for target, count in target_freq.items()}

    # Calculate q(w) - marginal probability of messages
    mssg_freq = Counter(messages)
    q_w = {msg: count / len(messages) for msg, count in mssg_freq.items()}

    # Calculate q(w|m) - probability distribution of messages given targets
    joint_counts = Counter(zip([tuple(m) for m in messages], targets))
    q_w_m = {k: v / target_freq[k[1]] for k, v in joint_counts.items()}

    # Calculate complexity: I(M; W)
    I_q_M_W = sum(
    p_m[m] * q_w_m[(w, m)] * math.log2((q_w_m[(w, m)] + 1e-12) / q_w[w])
    for (w, m) in joint_counts.keys())

    return I_q_M_W

def _get_need_probs(self, targets):
    target_freq = Counter(targets)
    return {target: count / len(targets) for target, count in target_freq.items()} 

def _calculate_information_loss(self, logs: core.Interaction):
    targets = [target_node for batch in logs.aux_input for target_node in batch['target_node']]
    need_probabilities = self._get_need_probs(targets)

    # Initialize dictionaries to store the sum of surprisals and the count of targets
    surprisal_sums = {target: 0.0 for target in need_probabilities}
    target_counts = {target: 0 for target in need_probabilities}

    log2 = torch.log(torch.tensor(2.0))

    # Iterate over each target and its corresponding log probabilities
    for i, target in enumerate(targets):
        if self.options.mode == 'gs':
            log_probs = logs.receiver_output[i].mean(dim=0)
        elif self.options.mode == 'rf':
            log_probs = logs.receiver_output[i]

        # Create a categorical distribution using the logits (log probabilities)
        dist = Categorical(logits=log_probs)

        # Calculate the surprisal of the target
        surprisal = -dist.logits[0].item() / log2.item()

        # Accumulate the surprisal sum for the current target
        surprisal_sums[target] += surprisal
        target_counts[target] += 1
    
    # Calculate the average surprisal for each target and the total information loss as a weighted sum of average surprisal
    avg_surprisal = {target: surprisal_sums[target] / target_counts[target] for target in surprisal_sums}
    information_loss = sum(need_probabilities[target] * avg_surprisal[target] for target in need_probabilities)
    
    return information_loss