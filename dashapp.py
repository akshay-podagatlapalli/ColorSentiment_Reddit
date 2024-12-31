from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

sent_data = pd.read_csv("output.csv")
emotion_data = pd.read_csv("output2.csv")

app = Dash()

app.layout = [
    html.H1(children='The Sentiments of Reddit', style={'textAlign':'center'}),
    dcc.Dropdown(sent_data.title.unique(), id='dropdown-selection'),
    dcc.Graph(id='graph-content')
]

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)

def update_graph(value):
    dff = sent_data[sent_data.country==value]
    return px.line(dff, x='year', y='pop')

if __name__ == '__main__':
    app.run(debug=True)