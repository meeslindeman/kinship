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

def plot_scatter(folder_path: str, y_metrics: list = ['information_loss', 'acc'], save: bool = False, mode: str = 'both'):
    dataframes, filenames = load_dataframes(folder_path)
    if not dataframes:
        print("No dataframes found.")
        return
    
    if mode not in ['train', 'test', 'both']:
        print("Invalid mode. Please choose 'train', 'test', or 'both'.")
        return

    titles_mapping = {
        'acc': 'Accuracy',
        'complexity': 'Complexity',
        'information_loss': 'Communicative Cost',
        'entropy': 'Accuracy (bits)',
        'kl_accuracy': 'Accuracy (bits)'
    }

    rows = len(y_metrics)
    cols = len(dataframes)

    max_marker_size = 20

    # Create a subplot
    fig = make_subplots(rows=rows, 
                        cols=cols, 
                        shared_xaxes=True,
                        shared_yaxes=True,
                        column_titles=[f"{filenames[i]}".rstrip('.csv') for i in range(cols)],
                        row_titles=[titles_mapping[metric] for metric in y_metrics],
                        horizontal_spacing=0.01, 
                        vertical_spacing=0.02,
                        x_title="complexity")

    for col, (df, filename) in enumerate(zip(dataframes, filenames), start=1):
        if mode != 'both':
            df = df[df["mode"] == mode]

        max_lens = sorted(df['max_len'].unique())
        dist = df['distractors'].unique()[0]

        max_epoch = df['epoch'].max()

        for row, metric in enumerate(y_metrics, start=1):
            for max_len in max_lens:
                df_max_len = df[df['max_len'] == max_len].dropna(subset=[metric])

                marker_sizes = max_marker_size * (df_max_len['epoch'] / max_epoch)

                trace = go.Scatter(x=df_max_len['complexity'], 
                                   y=df_max_len[metric], 
                                   mode='markers',
                                   marker=dict(size=marker_sizes, line=dict(width=1)),
                                   showlegend=False)
                
                fig.add_trace(trace, row=row, col=col)

    # Update layout
    fig.update_layout(legend=dict(
                        orientation="h",
                        xanchor="left",
                        yanchor="bottom",
                        x=0,
                        y=1),
                      height=600*rows,
                      width=800*cols)
    
    # Update layout
    fig.update_xaxes(range=[0, 7])
    for row, metric in enumerate(y_metrics, start=1):
        if metric == 'acc':
            fig.update_yaxes(range=[0, 1], row=row)
        else:
            fig.update_yaxes(range=[0, 4], row=row)
    
    if save:
        fig.write_image(f"plots/scatter_{'_'.join(y_metrics)}.png", scale=2)
    else:
        fig.show()

def plot_combined_scatter(folder_path: str, y_metric: str, save: bool = False, mode: str = 'both'):
    dataframes, filenames = load_dataframes(folder_path)
    if not dataframes:
        print("No dataframes found.")
        return
    
    if mode not in ['train', 'test', 'both']:
        print("Invalid mode. Please choose 'train', 'test', or 'both'.")
        return
    
    titles_mapping = {
        'acc': 'Accuracy',
        'complexity': 'Complexity',
        'information_loss': 'Communicative Cost',
        'entropy': 'Accuracy (bits)',
        'kl_accuracy': 'Accuracy (bits)'
    }

    max_marker_size = 20
    min_marker_size = 5

    # Create a subplot
    fig = make_subplots(rows=1, cols=1, 
                        shared_xaxes=True,
                        shared_yaxes=True,
                        horizontal_spacing=0.01, 
                        vertical_spacing=0.02,
                        x_title="complexity",
                        y_title=titles_mapping.get(y_metric, y_metric))

    for df, filename in zip(dataframes, filenames):
        if mode != 'both':
            df = df[df["mode"] == mode]

        max_lens = sorted(df['max_len'].unique())

        for max_len in max_lens:
            df_max_len = df[df['max_len'] == max_len].dropna(subset=[y_metric])

            marker_sizes = min_marker_size + (df_max_len['epoch'] / df_max_len['epoch'].max()) * (max_marker_size - min_marker_size)

            trace = go.Scatter(x=df_max_len['complexity'], 
                               y=df_max_len[y_metric], 
                               mode='markers',
                               marker=dict(size=marker_sizes, line=dict(width=1)),
                               name=f"{filename.rstrip('.csv')}")
                        
            fig.add_trace(trace, row=1, col=1)

    # Update layout
    fig.update_layout(legend=dict(
                        orientation="h",
                        xanchor="left",
                        yanchor="bottom",
                        x=0,
                        y=1),
                      height=600,
                      width=800)
    
    fig.update_xaxes(range=[0, 7])
    if y_metric == 'acc':
        fig.update_yaxes(range=[0, 4])
    else:
        fig.update_yaxes(range=[0, 4])
    
    if save:
        fig.write_image(f"plots/combined_scatter_{y_metric}.png", scale=2)
    else:
        fig.show()

# Usage
plot_combined_scatter('to_plot/', y_metric='kl_accuracy', save=True, mode='train')
#plot_scatter('to_plot/', y_metrics=['kl_accuracy'], save=True, mode='train')