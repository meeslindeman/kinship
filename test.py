import numpy as np
from collections import Counter

def calculate_entropy(probabilities):
    """Calculate Shannon entropy given a list of probabilities."""
    return -np.sum(p * np.log2(p) for p in probabilities if p > 0)

def calculate_complexity(messages):
    """Calculate complexity (Shannon entropy of messages)."""
    # Count frequency of each unique word
    message_counts = Counter(messages)
    total_messages = len(messages)
    # Calculate probabilities for each unique word
    probabilities = [count / total_messages for count in message_counts.values()]
    # Calculate entropy
    return calculate_entropy(probabilities)

def calculate_mutual_information(targets, messages, target_message_probs):
    """Calculate mutual information I(T; M) between targets and messages."""
    # Calculate P(T): target probabilities
    target_counts = Counter(targets)
    total_targets = len(targets)
    target_probs = {t: count / total_targets for t, count in target_counts.items()}
    
    # Calculate P(M): message probabilities
    message_counts = Counter(messages)
    total_messages = len(messages)
    message_probs = {m: count / total_messages for m, count in message_counts.items()}
    
    # Calculate conditional probabilities P(T | M) from target_message_probs
    conditional_entropy = 0
    for m, m_prob in message_probs.items():
        if m in target_message_probs:
            t_given_m_probs = target_message_probs[m]
            conditional_entropy += m_prob * calculate_entropy(t_given_m_probs.values())
    
    # Calculate H(T) and H(T | M)
    H_T = calculate_entropy(target_probs.values())
    H_T_given_M = conditional_entropy
    
    # Mutual information
    return H_T - H_T_given_M

def calculate_information_loss(targets, messages, target_message_probs):
    """Calculate information loss as H(T) - I(T; M)."""
    H_T = calculate_entropy([v / len(targets) for v in Counter(targets).values()])
    mutual_info = calculate_mutual_information(targets, messages, target_message_probs)
    return H_T - mutual_info

# Example Data
messages = ["dog", "cat", "dog", "bird"]
targets = ["A", "B", "A", "C"]

# Example target-message conditional probabilities P(T | M)
target_message_probs = {
    "dog": {"A": 0.8, "B": 0.1, "C": 0.1},
    "cat": {"A": 0.2, "B": 0.7, "C": 0.1},
    "bird": {"A": 0.1, "B": 0.1, "C": 0.8},
}

# Calculate Complexity (H(M))
complexity = calculate_complexity(messages)
print(f"Complexity (H(M)): {complexity:.4f}")

# Calculate Information Loss
info_loss = calculate_information_loss(targets, messages, target_message_probs)
print(f"Information Loss: {info_loss:.4f}")
