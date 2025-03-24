import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import plotly.colors as pc
import dash
from dash import dcc, html, Input, Output

# Import CSV
sentence_df = pd.read_csv("tech_topcis_df.csv")
network_df = sentence_df[~((sentence_df["Topic Name"] == "Nuclear Weapons") | (sentence_df["Topic Name"] == "Climate Change and Renewable Energy"))]

# Define groups
groups = {
    "ASEAN": ["Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam"],
    "Original_EU_plus": ["Belgium", "France", "Germany", "Italy", "Luxembourg", "Netherlands", "Austria", "Ireland"],
    "Baltic_Nordic_States": ["Estonia", "Latvia", "Lithuania", "Denmark", "Finland", "Sweden"],
    "Eastern_Europe": ["Poland", "Czech Republic", "Slovakia", "Hungary", "Romania", "Bulgaria"],
    "Southern_Europe": ["Portugal", "Spain", "Greece", "Cyprus", "Malta"],
    "West_Africa": ["Benin", "Burkina Faso", "Cape Verde", "Ivory Coast", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Liberia", "Mali", "Mauritania", "Niger", "Nigeria", "Senegal", "Sierra Leone", "Togo"],
    "Central_Africa": ["Cameroon", "Central African Republic", "Chad", "Congo", "Democratic Republic of the Congo", "Equatorial Guinea", "Gabon"],
    "East_Africa": ["Burundi", "Djibouti", "Eritrea", "Ethiopia", "Kenya", "Madagascar", "Malawi", "Mauritius", "Mozambique", "Rwanda", "Seychelles", "Somalia", "South Sudan", "Sudan", "Tanzania", "Uganda", "Zambia", "Zimbabwe"],
    "Southern_Africa": ["Angola", "Botswana", "Eswatini", "Lesotho", "Namibia", "South Africa"],
    "Middle_East": ["Afghanistan", "Bahrain", "Cyprus", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen"],
    "Mexico_Central_America": ["Mexico", "Guatemala", "Belize", "Honduras", "El Salvador", "Nicaragua", "Costa Rica", "Panama"],
    "Southern America": ["Colombia", "Ecuador", "Peru", "Bolivia", "Venezuela", "Argentina", "Chile", "Uruguay", "Paraguay"],
    "Caribbean": ["Antigua and Barbuda", "Bahamas", "Barbados", "Cuba", "Dominica", "Dominican Republic", "Grenada", "Haiti", "Jamaica", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago"],
    "Indian_Subcontinent": ["India", "Pakistan", "Bangladesh", "Nepal", "Sri Lanka", "Maldives", "Bhutan"],
    "Korea_Japan_Australia_NewZealand": ["South Korea", "North Korea", "Japan", "Australia", "New Zealand"]
}

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

group_dfs = {
    "ASEAN": ASEAN_df,
    "Original EU": Original_EU_df,
    "Baltic States": Baltic_Nordic_States_df,
    "Eastern Europe": Eastern_Europe_df,
    "Southern Europe": Southern_Europe_df,
    "West Africa": West_Africa_df,
    "Central Africa": Central_Africa_df,
    "East Africa": East_Africa_df,
    "Southern Africa": Southern_Africa_df,
    "Middle East": Middle_East_df,
    "Mexico Central America": Mexico_Central_America_df,
    "Southern America": Southern_America_df,
    "Caribbean": Caribbean_df,
    "Indian Subcontinent": Indian_Subcontinent_df,
    "Korea Japan Australia New Zealand": Korea_Japan_Australia_NewZealand_df
}

# Get the "Light24" color palette
light24_colors = pc.qualitative.Pastel

# Calculate topic trends
topic_trends = sentence_df.groupby(["Year", "Topic Name"]).size().reset_index(name="Mentions")
max_mentions_per_topic = topic_trends.groupby("Topic Name")["Mentions"].max()

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

# App layout
app.layout = html.Div(style={'font-family': 'Helvetica'}, children=[
    html.H3("Countries & Technology Mentions Network", style={"text-align": "left"}),
    group_selector,
    dcc.Graph(id="network-graph", config={'scrollZoom': True}, style={'flex': '1', 'height': '550px'}),
    html.Br(),
    dcc.Graph(id="highlighted-graph", config={'scrollZoom': True}, style={'flex': '1', 'height': '450px'}),
   
    html.H3("Total mentions of technology topics over time"),
    html.Div("Mentions by all countries are aggregated. Click on legends to turn on/off topics. Double click to select only that topic. Double click on the plot to reset zoom.",
     style={"text-align": "left","font-size": "14px"}),
    html.Br(),
    html.Div("Use the slider to filter in/out different topic by number of mentions.", style={"text-align": "left", "font-size": "14px"}),
    slider,
    
    dcc.Graph(id="topic-trends-graph")
])

# Callback function to update the network graph
@app.callback(
    [Output("network-graph", "figure"),
     Output("highlighted-graph", "figure")],
    [Input("group-selector", "value"),
     Input("network-graph", "clickData")]
)
def update_network_graph(group_name, click_data):
    dataframe = group_dfs[group_name]
    unique_countries = set(dataframe["Country Name"].unique())
    unique_topics = set(dataframe["Topic Name"].unique())
    edge_weights = dataframe.groupby(["Country Name", "Topic Name"]).size().reset_index(name="Weight")
    connected_countries = set(edge_weights["Country Name"])
    unique_countries = unique_countries.intersection(connected_countries)

    G = nx.Graph()
    G.add_nodes_from(unique_countries, bipartite=0)  # Countries
    G.add_nodes_from(unique_topics, bipartite=1)     # Topics

    for _, row in edge_weights.iterrows():
        G.add_edge(row["Country Name"], row["Topic Name"], weight=row["Weight"])

    pos = nx.kamada_kawai_layout(G)
    country_colors = {country: light24_colors[i % len(light24_colors)] for i, country in enumerate(unique_countries)}

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
            text=[f"{edge[0]} {edge[1]}: {weight} mentions"],
            mode='lines'
        )
        edge_traces.append(edge_trace)

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

    main_fig = go.Figure(data=edge_traces + [node_trace])
    main_fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        font=dict(family="Helvetica", size=12)
    )

    if click_data and "points" in click_data and click_data["points"]:
        selected_node = click_data["points"][0]["text"]
    else:
        selected_node = "United Kingdom"

    subgraph_edges = [(u, v) for u, v in G.edges(selected_node)]
    subgraph = G.edge_subgraph(subgraph_edges)
    highlighted_pos = {node: pos[node] for node in subgraph.nodes()}
    
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
            text=[f"{edge[0]} {edge[1]}: {weight} mentions"],
            mode='lines'
        )
        highlighted_edge_traces.append(edge_trace)

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
    highlighted_fig = go.Figure(data=highlighted_edge_traces + [highlighted_node_trace])
    highlighted_fig.update_layout(
        title=f"Highlighted: {selected_node} Connections",
        margin = dict(t = 100, l = 0),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, showticklabels=False, ticks='', zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, ticks='', zeroline=False),
        template="plotly_white",
        font=dict(family="Helvetica", size=12),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return main_fig, highlighted_fig

# Callback function to update the topic trends graph
@app.callback(
    Output("topic-trends-graph", "figure"),
    Input("max-mentions-slider", "value"))
def update_trends_graph(max_mentions):
    filtered_df = topic_trends[topic_trends["Topic Name"].isin(max_mentions_per_topic[(max_mentions_per_topic >= max_mentions[0]) & (max_mentions_per_topic <= max_mentions[1])].index)]
    sorted_topics = filtered_df.groupby("Topic Name")["Mentions"].max().sort_values(ascending=False).index
    pastel_palette = px.colors.qualitative.Pastel
    color_map = {topic: pastel_palette[i % len(pastel_palette)] for i, topic in enumerate(sorted_topics)}

    fig = px.line(
        filtered_df,
        x="Year",
        y="Mentions",
        color="Topic Name",
        category_orders={"Topic Name": sorted_topics},
        markers=True,
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
    app.run_server(debug=True)

