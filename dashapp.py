import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

# Load sentiment and emotion data
sent_data = pd.read_csv("data/output.csv")
emotion_data = pd.read_csv("data/output2.csv")

# Define a mapping of emotion types to specific colors
emotion_colors = {
    'joy': 'yellow',
    'sadness': 'blue',
    'anger': 'red',
    'fear': 'purple',
    'love': 'green',
    'surprise': 'orange'
}

# Normalize data
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

# Generate color gradient data for sentiment
def create_sentiment_gradient(post, scores):
    size = (750, 750)
    base_field = np.interp(np.linspace(0, len(scores) - 1, 750), np.arange(len(scores)), scores)
    noise = np.random.normal(loc=0.0, scale=5, size=size)
    Z = base_field + noise
    Z = gaussian_filter(Z, sigma=100)
    Z_norm = normalize_data(Z)
    return Z_norm

# Generate color gradient data for emotion
def create_emotion_gradient(post, scores, emotion_types):
    size = (750, 750)
    base_field = np.interp(np.linspace(0, len(scores) - 1, 750), np.arange(len(scores)), scores)
    noise = np.random.normal(loc=0.0, scale=5, size=size)
    Z = base_field + noise
    Z = gaussian_filter(Z, sigma=100)
    Z_norm = normalize_data(Z)
    return Z_norm

# Create a mapping between "Post 1", "Post 2", etc., and actual titles
post_titles = sent_data["title"].unique()
post_mapping = {f"Post {i+1}": post_titles[i] for i in range(len(post_titles))}

# App layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Sentiment and Emotion Visualization Dashboard"),
    dcc.Dropdown(
        id="title-dropdown",
        options=[{"label": f"Post {i+1}", "value": f"Post {i+1}"} for i in range(len(post_titles))],
        placeholder="Select a post",
        style={"width": "50%", "margin": "20px auto"}
    ),
    html.Div(
        id="post-title-container",
        style={"textAlign": "center", "margin": "20px 0"},
        children=html.H2(id="post-title", style={"fontWeight": "bold"})
    ),
    html.Div(
        id="plots-container",
        style={
            "display": "flex",
            "justify-content": "center",
            "align-items": "flex-start",
            "gap": "10px",
            "flex-wrap": "wrap"
        },
        children=[
            html.Div(
                style={"display": "flex", "flexDirection": "column", "alignItems": "center"},
                children=[
                    html.H2("Sentiment Gradient", style={"marginBottom": "0px"}),
                    dcc.Graph(id="sentiment-graph", style={"width": "750px", "height": "750px"})
                ]
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "column", "alignItems": "center"},
                children=[
                    html.H2("Emotion Gradient", style={"marginBottom": "0px"}),
                    dcc.Graph(id="emotion-graph", style={"width": "750px", "height": "750px"})
                ]
            )
        ]
    )
])

# Callbacks
@app.callback(
    [Output("post-title", "children"),
     Output("sentiment-graph", "figure"),
     Output("emotion-graph", "figure")],
    [Input("title-dropdown", "value")]
)
def update_graphs(selected_post):
    if not selected_post:
        return "", {}, {}  # Return empty if no title is selected

    # Get the actual post title from the mapping
    actual_title = post_mapping[selected_post]

    # Sentiment gradient
    sentiment_group_title = sent_data[sent_data["title"] == actual_title]
    sentiment_scores = sentiment_group_title["result"].to_numpy()
    sentiment_Z_norm = create_sentiment_gradient(actual_title, sentiment_scores)

    sentiment_fig = go.Figure()
    sentiment_fig.add_trace(
        go.Heatmap(
            z=sentiment_Z_norm,
            colorscale="Viridis",
            showscale=False
        )
    )
    sentiment_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        autosize=False,
        width=750,
        height=750
    )

    # Emotion gradient
    emotion_group_title = emotion_data[emotion_data["title"] == actual_title]
    emotion_scores = emotion_group_title["score"].to_numpy()
    emotion_types = emotion_group_title["type"].to_numpy()
    emotion_Z_norm = create_emotion_gradient(actual_title, emotion_scores, emotion_types)

    emotion_fig = go.Figure()
    emotion_fig.add_trace(
        go.Heatmap(
            z=emotion_Z_norm,
            colorscale="Jet",
            showscale=False
        )
    )
    emotion_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        autosize=False,
        width=750,
        height=750
    )

    return actual_title, sentiment_fig, emotion_fig


if __name__ == "__main__":
    app.run_server(debug=True)
