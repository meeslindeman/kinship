import os
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def load_dataframes(folder_path: str) -> list:
    dataframes = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            dataframes.append(df)
            filenames.append(file)
    return dataframes, filenames

def plot_metrics(folder_path: str, mode: str = 'train', save: bool = False):
    dataframes, filenames = load_dataframes(folder_path)
    if not dataframes:
        print("No dataframes found.")
        return
    
    if mode not in ['train', 'test', 'both']:
        print("Invalid mode. Please choose 'train', 'test', or 'both'.")
        return
    
    metrics = ['acc', 'complexity', 'information_loss']

    titles_mapping = {
        'acc': 'Accuracy',
        'complexity': 'Complexity',
        'information_loss': 'Communicative Cost',
        'entropy': 'Accuracy (bits)',
        'IB_bottleneck': 'IB Bottleneck'
    }

    rows = len(metrics)
    cols = len(dataframes)

    # Create a subplot
    fig = make_subplots(rows=rows, 
                        cols=cols, 
                        shared_yaxes=True,
                        shared_xaxes=True,
                        column_titles=[f"{filenames[i]}".rstrip('.csv') for i in range(cols)],
                        row_titles=[titles_mapping[metric] for metric in metrics],
                        horizontal_spacing=0.01, 
                        vertical_spacing=0.02,
                        x_title="Epoch",
                        y_title="Value")

    # Fill subplots
    for col, (df, filename) in enumerate(zip(dataframes, filenames), start=1):
        df_train = df[df["mode"] == "train"]
        df_test = df[df["mode"] == "test"]
        max_lens = sorted(df['max_len'].unique())
        dist = df['distractors'].unique()[0]

        for i, metric in enumerate(metrics):
            row = i + 1

            if mode in ['train', 'both']:
                for max_len in max_lens:
                    df_train_max_len = df_train[df_train['max_len'] == max_len].dropna(subset=[metric])
                    trace_train = go.Scatter(x=df_train_max_len['epoch'],
                                             y=df_train_max_len[metric],
                                             mode='lines+markers',
                                             name=f"Train",
                                             showlegend=(col == 1 and row == 1),
                                             marker=dict(size=4, line=dict(width=.5)),
                                             line=dict(width=2, color='#6495ED'))
                    fig.add_trace(trace_train, row=row, col=col)
            
            if mode in ['test', 'both']:
                for max_len in max_lens:
                    df_test_max_len = df_test[df_test['max_len'] == max_len].dropna(subset=[metric])
                    trace_test = go.Scatter(x=df_test_max_len['epoch'],
                                            y=df_test_max_len[metric],
                                            mode='lines+markers',
                                            name=f"Test",
                                            showlegend=(col == 1 and row == 1),
                                            marker=dict(size=4, line=dict(width=.5)),
                                            line=dict(width=2, color='#FFA500'))
                    fig.add_trace(trace_test, row=row, col=col)

    # Update layout
    fig.update_layout(legend=dict(
                        orientation="h",
                        xanchor="left",
                        yanchor="bottom",
                        x=0,
                        y=1),
                      height=300*rows,
                      width=400*cols)
    
    if save:
        fig.write_image("plots/experiments.png")
    else:
        fig.show()

# Usage
plot_metrics('to_plot/', mode='train', save=True)
