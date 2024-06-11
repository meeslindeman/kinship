import pandas as pd
import ast
import csv

# Function to load data from a CSV file
def load_data(df_path):
    """ Load data from CSV file at specified path. """
    return pd.read_csv(df_path)

def get_embeddings(df):
    """ Get embeddings from the DataFrame. """
    test_df = df[df["mode"] == "test"]
    if test_df.empty:
        return [], []

    last_row = test_df.iloc[-1]
    vectors = eval(last_row["messages"])  # Convert string to list
    targets = eval(last_row["target_node"])
    return vectors, targets

# Function to save vectors and metadata to TSV files
def save_to_tsv(vectors, targets, vector_file_path, metadata_file_path):
    # Flatten the list of vectors
    flattened_vectors = [vector for sublist in vectors for vector in sublist]
    
    # Write vectors to TSV file
    with open(vector_file_path, 'w') as f:
        for vector in flattened_vectors:
            f.write("\t".join(map(str, vector)) + "\n")
    
    # Write metadata to TSV file
    with open(metadata_file_path, 'w') as f:
        f.write("Target\n")  # Column header
        for target in targets:
            f.write(target + "\n")  # Write each target on a new line

df = load_data("results/single_dataframe_gs_1.csv")

# Get the embeddings from the DataFrame
embeddings, targets = get_embeddings(df)

# Save embeddings and metadata to TSV files
save_to_tsv(embeddings, targets, "messages.tsv", "metadata.tsv")

