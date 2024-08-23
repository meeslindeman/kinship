import plotly.express as px
import pandas as pd

def plot_experiment(df: pd.DataFrame, mode='both', save=True):
    colors = ["#FFA500", "#6495ED"]
    size = df["distractors"].iloc[0] + 1

    if mode != 'both':
        df = df[df['mode'] == mode]

    fig = px.line(df, x='epoch', y='acc', color='mode', color_discrete_sequence=colors)
    
    fig.update_traces(mode='markers+lines', marker=dict(size=4, line=dict(width=1)), line=dict(width=4))

    # Add a grey dotted line for chance level
    fig.add_hline(y=(1/size), line_dash="dash", line_color="grey", annotation_text="Chance Level")

    # Customizing the legend title
    fig.update_layout(legend_title_text='Mode')

    if save:
        # pip3 install kaleido
        fig.write_image(f"plots/single_accuracy_plot.png", scale=3)
    else:
        fig.show()

