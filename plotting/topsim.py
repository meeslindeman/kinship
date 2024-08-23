import os
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def load_dataframes_from_folder(folder_path: str) -> list:
    dataframes = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            dataframes.append(df)
            filenames.append(file)

    return dataframes, filenames

def plot_all_experiments_top_sim(folder_path: str, mode='both', save=False):
    colors = ["#FFA500", "#6495ED"]

    dfs, filenames = load_dataframes_from_folder(folder_path)
    if not dfs:
        print("No data found in the specified folder.")
        return

    # Extract options from filename
    option_a = sorted(set(f.split('_')[0] for f in filenames))
    option_b = sorted(set(f.split('_')[1].split('.')[0] for f in filenames), key=int)
    
    # Set the number of rows and columns
    rows = len(option_b)
    cols = len(option_a)

    # Create a subplot
    fig = make_subplots(rows=rows, 
                        cols=cols, 
                        shared_xaxes=True,
                        shared_yaxes=True,
                        column_titles=[f"Vocab size: {a}" for a in option_a],
                        row_titles=[f"Max len: {b}" for b in option_b],
                        horizontal_spacing=0.01, 
                        vertical_spacing=0.02,
                        x_title="Epoch",
                        y_title="Topographic Similarity")

    for df, filename in zip(dfs, filenames):
        if mode != 'both':
            df = df[df['mode'] == mode]

        # Add a plot for each DataFrame
        a, b = filename.split('_')
        b = b.split('.')[0]
        col = option_a.index(a) + 1
        row = option_b.index(b) + 1 

        for i, m in enumerate(['train', 'test']):
            sub_df = df[df['mode'] == m]
            color = colors[i]  

            trace = go.Scatter(x=sub_df['epoch'], 
                               y=sub_df['topsim'], 
                               mode='lines+markers', 
                               name=f'{m}', 
                               line=dict(color=color),
                               showlegend=True if (row, col) == (1, 1) else False)
            
            fig.add_trace(trace, row=row, col=col)
        fig.update_traces(mode='markers+lines', marker=dict(size=4, line=dict(width=1)), line=dict(width=4))

        # Check if 'topsim' data is available and add a line if it exists
        if 'topsim' in df.columns and not df['topsim'].isna().all():
            fig.add_hline(y=df['topsim'].mean(), line_dash="dash", line_color="grey", row=row, col=col, annotation_text="Average topsim")

    fig.update_yaxes(range=[0, 0.5])

    # Update layout
    fig.update_layout(legend=dict(
                        orientation="h",
                        xanchor="left",
                        yanchor="bottom",
                        x=0,
                        y=1),
                      height=400 * rows,
                      width=800 * cols)
    
    if save:
        fig.write_image("plots/topsim.png")
    else:
        fig.show()

# Usage
plot_all_experiments_top_sim('/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/results/no_age', mode='test', save=True)