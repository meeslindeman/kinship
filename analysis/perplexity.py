import math

# Perplexity of language corpus

freqs = {
    'mother': 318461, 'father': 299231, 'daughter': 120394, 'son': 188550, 
    'sister': 91290, 'brother': 118619, 'aunt': 27658, 'uncle': 43426,
    'grand': 59377, 'paternal': 1907, 'maternal': 6770,
    'niece': 5079, 'nephew': 7060, 'grandmother': 26372, 'grandfather': 21560, 
    'granddaughter': 4177, 'grandson': 6163, 'brother-in-law': 2987, 
    'sister-in-law': 2080, 'son-in-law': 2245, 'daughter-in-law': 1203,
}

freqs_primitives = {'female': 80946, 'male': 78716, 'child': 242330, 'older': 94860, 'younger': 56076}

total_rel = sum(freqs.values())

probabilities = {word: freq / total_rel for word, freq in freqs.items()}

perplexity = math.exp(-sum(p * math.log(p) for p in probabilities.values()))

# print(f'Total Relationship Frequencies: {total_rel}')
# print(f'Probabilities: {probabilities}')
print(f'Lexicon Perplexity: {perplexity}')

# Perplexity of agents:

from collections import defaultdict
from occurence.occ import load_data, analyze_co_occurrences

path = 'single_dataframe_gs'

df = load_data(f'results/{path}.csv')
combo_counts = analyze_co_occurrences(df)

# Function to count occurrences of each message
def count_mssg(combo_counts):
    """ Count total occurrences for each distinct message. """
    message_totals = defaultdict(int)
    sorted_combos = sorted(combo_counts.items(), key=lambda item: item[1], reverse=True)
    for combo, count in sorted_combos:
        message, _ = combo
        message_totals[message] += count
    return message_totals

message_counts = count_mssg(combo_counts)

total_mssgs = sum(message_counts.values())

mssg_probs = {mssg: count / total_mssgs for mssg, count in message_counts.items()}

mssg_perplexity = math.exp(-sum(p * math.log(p) for p in mssg_probs.values()))

# print(f'Total Messages: {total_mssgs}')
# print(f'Message Probabilities: {mssg_probs}')
print(f'Message Perplexity: {mssg_perplexity}')

