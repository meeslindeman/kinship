import pandas as pd
from collections import Counter, defaultdict
import ast

# Function to load data from a CSV file
def load_data(df_path):
    """ Load data from CSV file at specified path. """
    return pd.read_csv(df_path)

# Helper function to convert nested lists to tuples (for use as dict keys)
def to_tuple(data):
    """ Recursively convert list data to tuple data. """
    if isinstance(data, list):
        return tuple(to_tuple(item) for item in data)
    return data

# Analyze co-occurrences in a DataFrame
def analyze_co_occurrences(df):
    """ Analyze co-occurrences in the 'test' subset of the dataframe. """
    encoding = {
        1.0: 'Female', 2.0: 'Male', 3.0: 'Older', 4.0: 'Younger', 
        5.0: 'Equal', 6.0: 'Parent', 7.0: 'Child', 8.0: 'Spouse', 0.0: 'Padding'
    }

    test_df = df[df["mode"] == "test"]
    if test_df.empty:
        return Counter()

    last_row = test_df.iloc[-1]
    messages = ast.literal_eval(last_row['messages']) if isinstance(last_row['messages'], str) else last_row['messages']
    # sequences = ast.literal_eval(last_row['sequence']) if isinstance(last_row['sequence'], str) else last_row['sequence']
    sequences = ast.literal_eval(last_row['target_node']) if isinstance(last_row['target_node'], str) else last_row['target_node']

    combo_counter = Counter()
    for message, sequence in zip(messages, sequences):
        message = message[:-1]
        filtered_sequence = [item for item in sequence if item != 0.0]
        if not filtered_sequence:
            continue

        readable_sequence = [encoding.get(item, "Unknown") for item in filtered_sequence]
        combo_key_a = (tuple(message), tuple(readable_sequence))
        combo_key_b = (tuple(message), tuple(filtered_sequence))
        combo_key_c = (tuple(message), sequence)
        combo_counter[combo_key_c] += 1

    return combo_counter

# Function to print co-occurrence results
def print_co_occurrences(combo_counts):
    """ Print co-occurrence counts in a formatted manner. """
    print("Message and Sequence Co-occurrence Counts:")
    print("-----------------------------------------")
    total = 0
    sorted_combos = sorted(combo_counts.items(), key=lambda item: item[1], reverse=True)
    for combo, count in sorted_combos:
        message, sequence = combo
        total += count
        print(f"{message} and {sequence}: {count}")
    print("-----------------------------------------")
    print(f"Total number of co-occurrences: {total}")

# Function to print messages per category
def print_messages_per_category(combo_counts):
    """ Print all messages used per category with counts. """
    category_dict = defaultdict(Counter)
    for combo, count in combo_counts.items():
        message, sequence = combo
        category_dict[sequence][message] += count

    print("Messages per Category:")
    print("----------------------")
    for category, messages in category_dict.items():
        print(f"{category}:")
        for message, count in messages.items():
            print(f"Message: {message}, Count: {count}")
            print("\n")

# Function to count occurrences of each message
def count_mssg(combo_counts):
    """ Count total occurrences for each distinct message. """
    message_totals = defaultdict(int)
    sorted_combos = sorted(combo_counts.items(), key=lambda item: item[1], reverse=True)
    for combo, count in sorted_combos:
        message, _ = combo
        message_totals[message] += count
    
    print("Message Counts:")
    print("----------------------")
    for message, total in message_totals.items():
        print(f"Message {message}: {total} times")
    return message_totals

# Function to count occurrences of each symbol in messages
def count_symbols(combo_counts):
    """ Count total occurrences for each distinct symbol in messages. """
    symbol_totals = defaultdict(int)
    sorted_combos = sorted(combo_counts.items(), key=lambda item: item[1], reverse=True)
    for combo, count in sorted_combos:
        message, _ = combo
        for symbol in message:
            symbol_totals[symbol] += count
    for symbol, total in symbol_totals.items():
        print(f"Symbol {symbol}: {total} times")
    return symbol_totals

# Function to analyze attribute associations based on symbol
def analyze_attribute_associations(combo_counts, attribute_index, symbol=None):
    attribute_counts = defaultdict(int)
    encoding = {
        1.0: 'Female', 2.0: 'Male', 3.0: 'Older', 4.0: 'Younger', 
        5.0: 'Equal', 6.0: 'Parent', 7.0: 'Child', 8.0: 'Spouse', 0.0: 'Padding'
    }

    # Iterate over the message-sequence combinations and their counts
    for (message, sequence), count in combo_counts.items():
        # If a specific symbol is provided, check if it's in the message
        if symbol is not None and symbol not in message:
            continue

        # Extract the attribute from the sequence based on the provided index
        # Each sequence might contain multiple sets of attributes if length is more than 3
        for i in range(0, len(sequence), 3):  # Assuming each entity has three attributes
            if i + attribute_index < len(sequence):
                attribute = sequence[i + attribute_index]
                # Check and convert float to descriptive term if needed
                if attribute in encoding:
                    readable_attribute = encoding[attribute]
                    attribute_counts[readable_attribute] += count
                else:
                    attribute_counts[attribute] += count

    return attribute_counts

# Function to get unique symbols from combo_counts
def get_unique_symbols(combo_counts):
    """Extract all unique symbols used in messages from the combo_counts."""
    unique_symbols = set()
    for (message, _), _ in combo_counts.items():
        unique_symbols.update(message)
    return unique_symbols

# Function to analyze associations for all symbols and attributes
def analyze_all_symbols(combo_counts):
    """Analyze associations for all symbols and attributes."""
    unique_symbols = get_unique_symbols(combo_counts)
    all_relationship_counts = {}
    all_age_counts = {}
    all_gender_counts = {}

    for symbol in unique_symbols:
        # Compute the counts for each attribute category for this symbol
        relationship_counts = analyze_attribute_associations(combo_counts, 0, symbol)
        age_counts = analyze_attribute_associations(combo_counts, 1, symbol)
        gender_counts = analyze_attribute_associations(combo_counts, 2, symbol)

        # Store results for this symbol
        all_relationship_counts[symbol] = relationship_counts
        all_age_counts[symbol] = age_counts
        all_gender_counts[symbol] = gender_counts

    return all_relationship_counts, all_age_counts, all_gender_counts

# Function to invert the mapping from symbols to attributes
def invert_mapping(category_dict):
    inverted_dict = defaultdict(lambda: defaultdict(int))
    for symbol, attributes in category_dict.items():
        for attribute, count in attributes.items():
            inverted_dict[attribute][symbol] += count
    return inverted_dict
