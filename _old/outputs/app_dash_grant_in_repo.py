"""
This is the app that is in the public repo to embed it from posit cloud
""" 

import networkx as nx
import pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output
import plotly
import plotly.graph_objects as go


# Load data
df = pd.read_csv("Grant - OpenAlex/openalex_combined_dataset.csv")

# Introduce Dash app
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Country Collaboration Network by Topic and Year"),
    
    # Topic filter
    dcc.Dropdown(
        id="topic-filter",
        options=[
            {"label": "Artificial Intelligence", "value": "Artificial Intelligence"},
            {"label": "Engineering Biology", "value": "Engineering Biology"},
            {"label": "Quantum Technology", "value": "Quantum Technology"}
        ],
        value="Artificial Intelligence",
        clearable=False
    ),
    
    # Slider
    dcc.RangeSlider(
        id="year-slider",
        min=1993,
        max=2023,
        step=1,
        value=[1993, 2023],
        marks={year: str(year) for year in range(1993, 2024, 5)}
    ),
    
    # Graph for visualization
    dcc.Graph(id="country-network-graph")
])

# Topic and year range
@app.callback(
    Output("country-network-graph", "figure"),
    [Input("topic-filter", "value"), Input("year-slider", "value")]
)
def update_country_graph(selected_topic, selected_years):
    # Filter by topic and year range
    filtered_df = df[
        (df['Topic'] == selected_topic) & 
        (df['Year'] >= selected_years[0]) & 
        (df['Year'] <= selected_years[1])
    ]

    # Set country pairs manually
    country_edges = []
    for _, row in filtered_df.iterrows():
        countries = list(set(row['Institution Country'].split(', ')))
        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                country_edges.append((countries[i], countries[j]))
    
    # Manually count the country pairs
    country_pairs = {}
    for pair in country_edges:
        if pair in country_pairs:
            country_pairs[pair] += 1
        else:
            country_pairs[pair] = 1

    # Build the graph
    G_country = nx.Graph()
    for pair, weight in country_pairs.items():
        G_country.add_edge(pair[0], pair[1], weight=weight)

    # Generate positions for visualization
    pos_country = nx.spring_layout(G_country, seed=42)

    # Plotly figure
    edge_x = []
    edge_y = []
    for edge in G_country.edges():
        x0, y0 = pos_country[edge[0]]
        x1, y1 = pos_country[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_size = []
    node_text = []
    for node in G_country.nodes():
        x, y = pos_country[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(10 + 3 * G_country.degree(node))
        collaborations = sum(d['weight'] for _, _, d in G_country.edges(node, data=True))
        node_text.append(f"{node}<br>Collaborations: {collaborations}<br>Degree: {G_country.degree(node)}")

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="gray"),
        hoverinfo="none",
        mode="lines"
    )

    # Create nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            size=node_size,
            color="blue",
            line_width=2
        ),
        hoverinfo="text",
        text=node_text
    )

    # Setting up and combining into a figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Country Collaboration Network - {selected_topic} ({selected_years[0]}-{selected_years[1]})",
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)

