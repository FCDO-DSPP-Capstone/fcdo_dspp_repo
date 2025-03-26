
import os
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import time

import re

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import plotly.colors as pc


import dash
from dash import dcc, html, Input, Output



## 1. import csv 

sentence_df = pd.read_csv("Text Analysis Folder/data_out/tech_topcis_df.csv")

print(sentence_df["Year"].unique())

###
# grafico networks 

network_df = sentence_df[~((sentence_df["Topic Name"] == "Nuclear Weapons") | (sentence_df["Topic Name"] == "Climate Change and Renewable Energy"))]

# Define the groups
groups = {
    "ASEAN": ["Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam"],
    
    # European Union subgroups
    "Original_EU_plus": ["Belgium", "France", "Germany", "Italy", "Luxembourg", "Netherlands", "Austria", "Ireland"],
    "Baltic_Nordic_States": ["Estonia", "Latvia", "Lithuania", "Denmark", "Finland", "Sweden"],
    "Eastern_Europe": ["Poland", "Czech Republic", "Slovakia", "Hungary", "Romania", "Bulgaria"],
    "Southern_Europe": ["Portugal", "Spain", "Greece", "Cyprus", "Malta"],

    # African Union subgroups
    "West_Africa": ["Benin", "Burkina Faso", "Cape Verde", "Ivory Coast", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Liberia", "Mali", "Mauritania", "Niger", "Nigeria", "Senegal", "Sierra Leone", "Togo"],
    "Central_Africa": ["Cameroon", "Central African Republic", "Chad", "Congo", "Democratic Republic of the Congo", "Equatorial Guinea", "Gabon"],
    "East_Africa": ["Burundi", "Djibouti", "Eritrea", "Ethiopia", "Kenya", "Madagascar", "Malawi", "Mauritius", "Mozambique", "Rwanda", "Seychelles", "Somalia", "South Sudan", "Sudan", "Tanzania", "Uganda", "Zambia", "Zimbabwe"],
    "Southern_Africa": ["Angola", "Botswana", "Eswatini", "Lesotho", "Namibia", "South Africa"],
    
    # Middle East
    "Middle_East": ["Afghanistan", "Bahrain", "Cyprus", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen"],

    # Latin America & Caribbean subgroups
    "Mexico_Central_America": ["Mexico", "Guatemala", "Belize", "Honduras", "El Salvador", "Nicaragua", "Costa Rica", "Panama"],
    "Southern America": ["Colombia", "Ecuador", "Peru", "Bolivia", "Venezuela", "Argentina", "Chile", "Uruguay", "Paraguay"],
    "Caribbean": ["Antigua and Barbuda", "Bahamas", "Barbados", "Cuba", "Dominica", "Dominican Republic", "Grenada", "Haiti", "Jamaica", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago"],

    # Indian Subcontinent
    "Indian_Subcontinent": ["India", "Pakistan", "Bangladesh", "Nepal", "Sri Lanka", "Maldives", "Bhutan"],

    # Korea, Japan, Australia, New Zealand
    "Korea_Japan_Australia_NewZealand": ["Korea, Republic of", "Korea, Democratic People's Republic of", "Japan", "Australia", "New Zealand"]
}

# Always include these major countries
always_include = ["United Kingdom", "United States", "China"]

# Function to create a filtered DataFrame for each group
def filter_by_group(group_name):
    return network_df[network_df["Country Name"].isin(groups[group_name] + always_include)]

# Creating separate DataFrames for each subgroup
ASEAN_df = filter_by_group("ASEAN")
Original_EU_df = filter_by_group("Original_EU_plus")
Baltic_Nordic_States_df = filter_by_group("Baltic_Nordic_States")
Eastern_Europe_df = filter_by_group("Eastern_Europe")
Southern_Europe_df = filter_by_group("Southern_Europe")

West_Africa_df = filter_by_group("West_Africa")
Central_Africa_df = filter_by_group("Central_Africa")
East_Africa_df = filter_by_group("East_Africa")
Southern_Africa_df = filter_by_group("Southern_Africa")

Middle_East_df = filter_by_group("Middle_East")

Mexico_Central_America_df = filter_by_group("Mexico_Central_America")
Southern_America_df = filter_by_group("Southern America")
Caribbean_df = filter_by_group("Caribbean")

Indian_Subcontinent_df = filter_by_group("Indian_Subcontinent")

Korea_Japan_Australia_NewZealand_df = filter_by_group("Korea_Japan_Australia_NewZealand")




# List of DataFrames to iterate through, using their group names
group_dfs = {
    "ASEAN": ASEAN_df,
    "Original EU": Original_EU_df,
    "Baltic States": Baltic_Nordic_States_df,  # Corrected to match the variable name
    "Eastern Europe": Eastern_Europe_df,
    "Southern Europe": Southern_Europe_df,
    "West Africa": West_Africa_df,
    "Central Africa": Central_Africa_df,
    "East Africa": East_Africa_df,
    "Southern Africa": Southern_Africa_df,
    "Middle East": Middle_East_df,
    "Mexico Central America": Mexico_Central_America_df,
    "Southern America": Southern_America_df,  # Corrected to match the variable name
    "Caribbean": Caribbean_df,
    "Indian Subcontinent": Indian_Subcontinent_df,  # Added the Indian Subcontinent group
    "Korea Japan Australia New Zealand": Korea_Japan_Australia_NewZealand_df  # Added Korea, Japan, Australia, New Zealand
}


# Get the "Light24" color palette
light24_colors = pc.qualitative.Pastel


# Create Dash App
app = dash.Dash(__name__)

# Create group selector
group_selector = dcc.Dropdown(
    id="group-selector",
    options=[{"label": group_name, "value": group_name} for group_name in group_dfs.keys()],
    value=list(group_dfs.keys())[0],
    clearable=False,
    style={"font-family": "Helvetica"}
)

# App layout
app.layout = html.Div([
    html.H1("Countries & Technology Mentions Network", style={"text-align": "center", 'font-family': 'Helvetica'}),
    group_selector,
    dcc.Graph(id="network-graph", config={'scrollZoom': True}),
    html.Br(),
    dcc.Graph(id="highlighted-graph", config={'scrollZoom': True})
])

# Callback function to update the graph
@app.callback(
    [Output("network-graph", "figure"),
     Output("highlighted-graph", "figure")],
    [Input("group-selector", "value"),
     Input("network-graph", "clickData")]
)
def update_graph(group_name, click_data):
    print(f"Generating network graph for: {group_name}")

    # Get the DataFrame for the selected group
    dataframe = group_dfs[group_name]

    # Get unique countries and topics
    unique_countries = set(dataframe["Country Name"].unique())
    unique_topics = set(dataframe["Topic Name"].unique())

    # Aggregate edge weights
    edge_weights = dataframe.groupby(["Country Name", "Topic Name"]).size().reset_index(name="Weight")

    # Filter out countries that do not have topic connections
    connected_countries = set(edge_weights["Country Name"])
    unique_countries = unique_countries.intersection(connected_countries)

    # Create Graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(unique_countries, bipartite=0)  # Countries
    G.add_nodes_from(unique_topics, bipartite=1)    # Topics

    # Add edges with weights
    for _, row in edge_weights.iterrows():
        G.add_edge(row["Country Name"], row["Topic Name"], weight=row["Weight"])

    # Node positioning
    pos = nx.kamada_kawai_layout(G)

    # Assign each country a unique color
    country_colors = {country: light24_colors[i % len(light24_colors)] for i, country in enumerate(unique_countries)}

    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]["weight"]
        line_width = max(1, weight * 0.5)  

        edge_color = country_colors.get(edge[0], "darkgrey")

        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=line_width, color=edge_color),
            hoverinfo='text',
            text=[f"{edge[0]} ↔ {edge[1]}: {weight} mentions"],
            mode='lines'
        )
        edge_traces.append(edge_trace)

    # Create node traces
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(country_colors.get(node, "darkgrey"))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=20, color=node_color, line=dict(width=2, color='black')),
        text=node_text,
        textposition="top center",
        hoverinfo="text"

    )

    # Create the main figure
    main_fig = go.Figure(data=edge_traces + [node_trace])
    main_fig.update_layout(
        title=f"{group_name} Countries & Technology Mentions Network",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        font=dict(family="Helvetica", size=12)
    )

    # Create highlighted subgraph
    if click_data and "points" in click_data and click_data["points"]:
        selected_node = click_data["points"][0]["text"]
        subgraph_edges = [(u, v) for u, v in G.edges(selected_node)]
        subgraph = G.edge_subgraph(subgraph_edges)
        highlighted_pos = {node: pos[node] for node in subgraph.nodes()}
        
        # Create highlighted edge traces
        highlighted_edge_traces = []
        for edge in subgraph.edges(data=True):
            x0, y0 = highlighted_pos[edge[0]]
            x1, y1 = highlighted_pos[edge[1]]
            weight = edge[2]["weight"]
            line_width = max(1, weight * 0.5)

            edge_color = country_colors.get(edge[0], "darkgrey")

            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=line_width, color=edge_color),
                hoverinfo='text',
                text=[f"{edge[0]} ↔ {edge[1]}: {weight} mentions"],
                mode='lines'
            )
            highlighted_edge_traces.append(edge_trace)

        # Create highlighted node traces
        highlighted_node_x, highlighted_node_y, highlighted_node_text, highlighted_node_color = [], [], [], []
        for node in subgraph.nodes():
            x, y = highlighted_pos[node]
            highlighted_node_x.append(x)
            highlighted_node_y.append(y)
            highlighted_node_text.append(node)
            highlighted_node_color.append(country_colors.get(node, "darkgrey"))

        highlighted_node_trace = go.Scatter(
            x=highlighted_node_x, y=highlighted_node_y,
            mode='markers+text',
            marker=dict(size=20, color=highlighted_node_color, line=dict(width=2, color='black')),
            text=highlighted_node_text,
            textposition="top center",
            hoverinfo="text"
        )

        # Create the highlighted figure
        highlighted_fig = go.Figure(data=highlighted_edge_traces + [highlighted_node_trace])
        highlighted_fig.update_layout(
            title=f"Highlighted: {selected_node} Connections",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            font=dict(family="Helvetica", size=12)

        )
    else:
        highlighted_fig = go.Figure()

    return main_fig, highlighted_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


############################################
#### visuazlaition 2:
#### visuazlaition of topic mentions counts by year (not country)
####################################


# Calculate topic trends
topic_trends = sentence_df.groupby(["Year", "Topic Name"]).size().reset_index(name="Mentions")

# Calculate the max mentions per topic
max_mentions_per_topic = topic_trends.groupby("Topic Name")["Mentions"].max()

app = dash.Dash(__name__)
# Create a range slider for filtering the topics by max count
slider_marks = {i: f"{i}" for i in range(0, max_mentions_per_topic.max() + 1, 50)}
slider = dcc.RangeSlider(
    id="max-mentions-slider",
    marks=slider_marks,
    min=0,
    max=max_mentions_per_topic.max(),
    step=1,
    value=[0, max_mentions_per_topic.max()]
)

# Create the app layout
app.layout = html.Div([
    html.H1("Trends in Technology Topics Over Time"),
    slider,
    dcc.Graph(id="topic-trends-graph")
])

# Define the callback function
@app.callback(
    Output("topic-trends-graph", "figure"),
    Input("max-mentions-slider", "value"))
def update_graph(max_mentions):
    # Filter based on slider input
    filtered_df = topic_trends[topic_trends["Topic Name"].isin(max_mentions_per_topic[max_mentions_per_topic <= max_mentions[1]].index)]

    # Sort topics
    sorted_topics = filtered_df.groupby("Topic Name")["Mentions"].max().sort_values(ascending=False).index

    # Define colors using Plotly's color palette
    pastel_palette = px.colors.qualitative.Pastel
    color_map = {topic: pastel_palette[i % len(pastel_palette)] for i, topic in enumerate(sorted_topics)}

    # Plot using filtered data
    fig = px.line(
        filtered_df,
        x="Year",
        y="Mentions",
        color="Topic Name",
        category_orders={"Topic Name": sorted_topics},
        markers=True,
        title="Trends in Technology Topics Over Time",
        labels={"Mentions": "Number of Mentions"},
        template="plotly_white",
        color_discrete_map=color_map
    )

    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=1),
        showlegend=True,
        font=dict(family="Helvetica")
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

############################################
#### visuazlaition 3:
#### Viz of countries cisne similarity embeddings 
####################################

# Define topic-to-group mapping by topic numbers
topic_groups = {
    "Military Technology": [0, 1, 2],  # Nuclear Weapons, Biological Weapons, Chemical Weapons
    "Dual Use Technology": [3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 12, 13, 14, 15, 16, 17, 18],  # AI, Quantum Tech, etc.
    "Civilian Technology": [11, 12, 13, 14, 15, 16, 17, 18],  # Climate change, Electric cars, etc.
    "Capacity Building": [16, 17, 18],  # Digital Divide, Tech Capacity Building, etc.
    "Grey Zone": [19, 20, 21]  # Cyberattacks, Election Interference, etc.
}

# Function to assign group based on topic number(s)
def assign_group_by_number(topics):
    # If 'topics' is a single integer, convert it into a list
    if isinstance(topics, int):
        topics = [topics]  # Make it iterable

    # Check which group the topics belong to
    for group, topic_list in topic_groups.items():
        if any(topic in topic_list for topic in topics):
            return group
    return "Uncategorized"  # Default if no match found

# Apply function to create "group" column based on topic numbers
sentence_df["group"] = sentence_df["Topic"].apply(assign_group_by_number)

# Print names of countries that contain "Korea"
countries_with_korea = [
    str(country).lower() for country in sentence_df['Country Name'].unique()
    if 'Korea' in str(country)
]