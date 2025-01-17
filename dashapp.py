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
def create_sentiment_gradient(post, scores, comments):
    size = (750, 750)
    base_field = np.interp(np.linspace(0, len(scores) - 1, 750), np.arange(len(scores)), scores)
    noise = np.random.normal(loc=0.0, scale=5, size=size)
    Z = base_field + noise
    Z = gaussian_filter(Z, sigma=100)
    Z_norm = normalize_data(Z)
    
    # Create a custom data array with interpolated comments
    interpolated_indices = np.linspace(0, len(comments) - 1, 750, dtype=int)
    customdata = np.array([comments[idx] for idx in interpolated_indices]).reshape(750, 1).repeat(750, axis=1)
    
    return Z_norm, customdata


# Generate color gradient data for emotion
def create_emotion_gradient(post, scores, comments):
    size = (750, 750)
    base_field = np.interp(np.linspace(0, len(scores) - 1, 750), np.arange(len(scores)), scores)
    noise = np.random.normal(loc=0.0, scale=5, size=size)
    Z = base_field + noise
    Z = gaussian_filter(Z, sigma=100)
    Z_norm = normalize_data(Z)
    
    # Create a custom data array with interpolated comments
    interpolated_indices = np.linspace(0, len(comments) - 1, 750, dtype=int)
    customdata = np.array([comments[idx] for idx in interpolated_indices]).reshape(750, 1).repeat(750, axis=1)
    
    return Z_norm, customdata

# Create a mapping between "Post 1", "Post 2", etc., and actual titles
post_titles = sent_data["title"].unique()
post_mapping = {f"Post {i+1}": post_titles[i] for i in range(len(post_titles))}

# App layout
app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f9f9f9",
        "padding": "20px"
    },
    children=[
        # Dashboard Title
        html.H1(
            "Sentiment and Emotion Visualization Dashboard",
            style={
                "textAlign": "center",
                "color": "#333",
                "marginBottom": "20px"
            }
        ),
        
        # Dropdown for post selection
        dcc.Dropdown(
            id="title-dropdown",
            options=[
                {"label": f"Post {i+1}", "value": f"Post {i+1}"}
                for i in range(len(post_titles))
            ],
            placeholder="Select a post",
            style={
                "width": "60%",
                "margin": "0 auto 20px auto",
                "padding": "10px",
                "borderRadius": "5px",
                "border": "1px solid #ccc"
            }
        ),
        
        # Post title container
        html.Div(
            id="post-title-container",
            style={
                "textAlign": "center",
                "margin": "20px 0"
            },
            children=html.H2(
                id="post-title",
                style={
                    "fontWeight": "bold",
                    "color": "#444"
                }
            )
        ),
        
        # Graphs container
        html.Div(
            id="plots-container",
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "flex-start",
                "gap": "20px",
                "flexWrap": "wrap",
                "margin": "0 auto",
                "padding": "10px"
            },
            children=[
                # Sentiment graph
                html.Div(
                    style={
                        "backgroundColor": "#fff",
                        "padding": "15px",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "width": "45%"
                    },
                    children=[
                        html.H2(
                            "Sentiment Gradient",
                            style={
                                "marginBottom": "10px",
                                "color": "#555"
                            }
                        ),
                        dcc.Graph(
                            id="sentiment-graph",
                            style={
                                "width": "100%",
                                "height": "100%",
                                "maxWidth": "750px",
                                "maxHeight": "750px",
                                "boxSizing": "border-box"}
                        ),
                        html.Div(
                            id="sentiment-comment-box",
                            style={
                                "marginTop": "20px",
                                "padding": "10px",
                                "border": "1px solid #ddd",
                                "borderRadius": "5px",
                                "backgroundColor": "#f9f9f9",
                                "fontSize": "16px",
                                "fontFamily": "Arial",
                                "minHeight": "50px",
                                "width": "100%"
                            },
                            children="Click on a section of the plot to see the comment here."
                        )
                    ]
                ),
                
                # Emotion graph
                html.Div(
                    style={
                        "backgroundColor": "#fff",
                        "padding": "15px",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "width": "45%"
                    },
                    children=[
                        html.H2(
                            "Emotion Gradient",
                            style={
                                "marginBottom": "10px",
                                "color": "#555"
                            }
                        ),
                        dcc.Graph(
                                id="emotion-graph",
                                style={
                                    "width": "100%",
                                    "height": "100%",
                                    "maxWidth": "750px",
                                    "maxHeight": "750px",
                                    "boxSizing": "border-box"
                                }
                        ),
                        html.Div(
                            id="emotion-comment-box",
                            style={
                                "marginTop": "20px",
                                "padding": "10px",
                                "border": "1px solid #ddd",
                                "borderRadius": "5px",
                                "backgroundColor": "#f9f9f9",
                                "fontSize": "16px",
                                "fontFamily": "Arial",
                                "minHeight": "50px",
                                "width": "100%"
                            },
                            children="Click on a section of the plot to see the comment here.")
                    ]
                )
            ]
        )
    ]
)


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
    sentiment_comments = sentiment_group_title["comment"].to_numpy()
    sentiment_Z_norm, sentiment_customdata = create_sentiment_gradient(actual_title, sentiment_scores, sentiment_comments)

    # Sentiment Heatmap
    sentiment_fig = go.Figure()
    sentiment_fig.add_trace(
        go.Heatmap(
            z=sentiment_Z_norm,
            customdata=sentiment_customdata,
            hovertemplate=(
                "<span style='font-size:16px; font-family:sans-serif; word-wrap:break-word;'>"
                "Comment: %{customdata}"
                "</span><extra></extra>"
            ),
            colorscale="Viridis",
            showscale=True,  # Display the colorbar
            colorbar=dict(
                title="Sentiment",  # Label for the colorbar
                tickvals=[0, 0.5, 1],  # Position of ticks
                ticktext=["Negative", "Neutral", "Positive"],  # Custom labels
                ticks="outside",
                ticklen=5,
                thickness=15,
                len=0.8,
                x=1.05
            )
        )
    )
    sentiment_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # Emotion Heatmap
    # Define custom colorscale
    emotion_colors = {
        'joy': 'yellow',
        'sadness': 'blue',
        'anger': 'red',
        'fear': 'purple',
        'love': 'green',
        'surprise': 'orange'
    }
    emotion_colorscale = [
        [0.0, emotion_colors['sadness']],
        [0.2, emotion_colors['fear']],
        [0.4, emotion_colors['anger']],
        [0.6, emotion_colors['love']],
        [0.8, emotion_colors['joy']],
        [1.0, emotion_colors['surprise']]
    ]
    # Emotion gradient
    emotion_fig = go.Figure()
    emotion_fig.add_trace(
        go.Heatmap(
            z=sentiment_Z_norm,
            customdata=sentiment_customdata,
            hovertemplate=(
                "<span style='font-size:16px; font-family:sans-serif; word-wrap:break-word;'>"
                "Comment: %{customdata}"
                "</span><extra></extra>"
            ),
            colorscale=emotion_colorscale,
            showscale=True,  # Display the colorbar
            colorbar=dict(
                title="Emotion",
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["Sadness", "Fear", "Anger", "Love", "Joy", "Surprise"],
                ticks="outside",
                ticklen=5,
                thickness=15,
                len=0.8,
                x=1.05
            )
        )
    )
    emotion_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return actual_title, sentiment_fig, emotion_fig

@app.callback(
    Output("sentiment-comment-box", "children"),
    [Input("sentiment-graph", "clickData")]
)
def update_sentiment_comment(clickData):
    if clickData and "points" in clickData:
        return clickData["points"][0]["customdata"]
    return "Click on a section of the plot to see the comment here."

@app.callback(
    Output("emotion-comment-box", "children"),
    [Input("emotion-graph", "clickData")]
)
def update_emotion_comment(clickData):
    if clickData and "points" in clickData:
        return clickData["points"][0]["customdata"]
    return "Click on a section of the plot to see the comment here."

if __name__ == "__main__":
    app.run_server(debug=True)
