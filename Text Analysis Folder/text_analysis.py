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


"""
This code is going to compute The Cosine Similarity evolution over time of diferent countries
against each other 

Not all countries are listed because they have to have neough mentions of techology 
on their speeches to appear all the years. 

To calcualte the change over time, we need to define a base year of referencem which is 2021, that is fiexed,
and a reference country agains all the other varies closer or farther apart, where 1 is more 
similar and 0 more different,
The referenffce countries does not appear as a Line on the grpah. 
The reference country for selection simplicity is pre defined to an option between the UK, the US and China.

Finally for computatioanl efficency and faccilitate anaylisis, the user needs to select a group of countries to 
be compared to the reference country, and a Tech toopic group of interest.

The final code structure is then:

    - Original sentence level Dataframe, adding geogrpahic group column
    - Add new macro topic column to sentence level dataframe
    - Create collapsed data frame with the average of each country by macro topic.
        - This collapsed dataframe is the result of grouping rows by country, topic_group and year
            and computign the average of its embeddings in a -+2 years window
    - the final dataframe has then one row per country-year-topic_group combination, and has also the column
        with the country_group and the average embedding.

    - With this dataframe then, we make the interactive application. IN this application, we do this steps: 

        1. Select a topic group
        2. Select a reference country
        3. Select a country group
        4. Calculate the cosine similarity between the reference country and each
         country in the country group for each year.
        year 2021. 
        5. Calculate the average cosine similarity of all the countries for each year.
        6. Plot a line graph of all countries and the average as a black line
         with years in X values and cosine_similarity values in Y values
"""


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import plotly.colors as pc
import dash
from dash import dcc, html, Input, Output

sentence_df = pd.read_csv("Text Analysis Folder/data_out/tech_topcis_df.csv")
embeddings = np.load("Text Analysis Folder/data_out/tech_embeddings.npy")

topic_groups = {
    "Military Technology": [0, 1, 2],  # Nuclear Weapons, Biological Weapons, Chemical Weapons
    "Dual Use Technology": [3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 12, 13, 14, 15, 16, 17, 18],  # AI, Quantum Tech, etc.
    "Civilian Technology": [11, 12, 13, 14, 15, 16, 17, 18],  # Climate change, Electric cars, etc.
}

country_groups = {
    "ASEAN": ["Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam", "India", "Pakistan"],
    "European Union": ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"],
    "African Union": ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo", "Democratic Republic of the Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"],
    "Middle East": ["Afghanistan", "Bahrain", "Cyprus", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen"],
    "Latin America and Caribbean": ["Antigua and Barbuda", "Argentina", "Bahamas", "Barbados", "Belize", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "Ecuador", "El Salvador", "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago", "Uruguay", "Venezuela"],
    "Reference": ["United States", "United Kingdom", "China"]
}

# Add geographic group column
def assign_country_group(country):
    for group, countries in country_groups.items():
        if country in countries:
            return group
    return None

sentence_df['Country Group'] = sentence_df['Country Name'].apply(assign_country_group)

# Add macro topic column
def assign_topic_group(topic_id):
    for group_name, topic_ids in topic_groups.items():
        if topic_id in topic_ids:
            return group_name
    return None

sentence_df['Macro Topic'] = sentence_df['Topic'].apply(assign_topic_group)

# Ensure 'Year' is an integer
sentence_df['Year'] = sentence_df['Year'].astype(int)

# Group by rows and compute average embeddings
def average_embeddings_five_year_window(df, embeddings, group_vars):
    # Add an index column to the dataframe to keep track of original indices
    df['index'] = np.arange(len(df))
    
    # Initialize lists to store the grouped keys and average embeddings
    grouped_keys = []
    avg_embeddings_list = []
    
    # Get the unique years in the dataframe
    unique_years = df['Year'].unique()
    
    # Iterate over each unique combination of ['Country Name', 'Country Group', 'Macro Topic']
    for name, group in df.groupby(group_vars[:-1]):
        for year in unique_years:
            # Define the five-year window
            window = range(year - 3, year + 4)
            
            # Get the indices of the rows within the five-year window
            window_indices = group[group['Year'].isin(window)].index
            
            if len(window_indices) > 0:
                # Compute the average embedding for the current group and window
                avg_embedding = embeddings[window_indices].mean(axis=0)
                
                # Append the group keys and year to the grouped_keys list
                grouped_keys.append((*name, year))
                avg_embeddings_list.append(avg_embedding)
    
    # Convert the grouped_keys list to a dataframe
    grouped_keys_df = pd.DataFrame(grouped_keys, columns=group_vars)
    grouped_keys_df = grouped_keys_df.sort_values(by=['Year'])
    # Convert the avg_embeddings_list to a numpy array
    avg_embeddings_array = np.vstack(avg_embeddings_list)
    
    return grouped_keys_df, avg_embeddings_array

group_vars = ['Country Name', 'Country Group', 'Macro Topic', 'Year']

grouped_keys, avg_embeddings = average_embeddings_five_year_window(sentence_df, embeddings, group_vars)

print(grouped_keys)
print(avg_embeddings)




def calculate_cosine_similarity(grouped_keys, avg_embeddings):
    # Initialize an empty list to store the results
    results = []

    # Get the unique macro topics and reference countries
    macro_topics = grouped_keys['Macro Topic'].unique()
    reference_countries = grouped_keys[grouped_keys['Country Group'] == 'Reference']['Country Name'].unique()

    # Iterate over each macro topic
    for topic in macro_topics:
        # Iterate over each reference country
        for ref_country in reference_countries:
            # Get the reference embedding for the selected reference country, macro topic, and year 2021
            reference_embedding = avg_embeddings[(grouped_keys['Country Name'] == ref_country) & 
                                                 (grouped_keys['Macro Topic'] == topic) & 
                                                 (grouped_keys['Year'] == 2021)][0]
            
            # Filter the DataFrame based on the macro topic
            topic_df = grouped_keys[grouped_keys['Macro Topic'] == topic]
            
            # Iterate over each country in the topic DataFrame
            for country in topic_df['Country Name'].unique():
                country_df = topic_df[topic_df['Country Name'] == country]
                
                # Iterate over each year for the current country
                for year in country_df['Year'].unique():
                    country_embedding = avg_embeddings[(grouped_keys['Country Name'] == country) & 
                                                       (grouped_keys['Macro Topic'] == topic) & 
                                                       (grouped_keys['Year'] == year)][0]
                    similarity = cosine_similarity([reference_embedding], [country_embedding])[0][0]
                    
                    # Append the result to the list
                    results.append({
                        'Country Name': country,
                        'Country Group': country_df['Country Group'].iloc[0],
                        'Macro Topic': topic,
                        'Year': year,
                        'Reference': ref_country,
                        'Similarity': similarity
                    })
    
    # Create a DataFrame from the results
    similarity_df = pd.DataFrame(results)
    return similarity_df.sort_values(by=['Country Name', 'Macro Topic', 'Year'])

similarity_df = calculate_cosine_similarity(grouped_keys, avg_embeddings)



# Dash application setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("Country Speeches Similarity Analysis", style={'font-family': 'Helvetica'}),
    
    html.Label("Select a Topic Group", style={'font-family': 'Helvetica'}),
    dcc.Dropdown(
        id='topic-group-dropdown',
        options=[{'label': topic, 'value': topic} for topic in similarity_df['Macro Topic'].unique()],
        style={'font-family': 'Helvetica'}
    ),

    html.Br(),
    
    html.Label("Select a Country Group", style={'font-family': 'Helvetica'}),
    dcc.Dropdown(
        id='country-group-dropdown',
        options=[{'label': group, 'value': group} for group in similarity_df['Country Group'].unique()],
        style={'font-family': 'Helvetica'}
    ),
    
    html.Br(),

    
    html.Label("Select a Reference Country", style={'font-family': 'Helvetica'}),
    dcc.Dropdown(
        id='reference-country-dropdown',
        style={'font-family': 'Helvetica'}
    ),
    
    dcc.Graph(id='similarity-graph', style={'font-family': 'Helvetica'})
])

@app.callback(
    Output('reference-country-dropdown', 'options'),
    Input('country-group-dropdown', 'value')
)
def set_reference_country_options(selected_group):
    filtered_df = similarity_df[similarity_df['Country Group'] == 'Reference']
    reference_countries = filtered_df['Country Name'].unique()
    options = [{'label': country, 'value': country} for country in reference_countries]
    return options


@app.callback(
    Output('similarity-graph', 'figure'),
    Input('topic-group-dropdown', 'value'),
    Input('reference-country-dropdown', 'value'),
    Input('country-group-dropdown', 'value')
)
def update_graph(topic_group, reference_country, country_group):
    # Filter the DataFrame based on the selected topic group and country group
    filtered_df = similarity_df[(similarity_df['Macro Topic'] == topic_group) & (similarity_df['Country Group'] == country_group)]

    # Further filter based on the selected reference country
    filtered_df = filtered_df[filtered_df['Reference'] == reference_country]

    # Plot the results using Plotly Express
    fig_height = 800
    if filtered_df.empty:
        fig = {}
    else:
        fig = px.scatter(
            filtered_df, 
            x='Year', 
            y='Similarity', 
            color='Country Name', 
            title=f'Similarity of {country_group} against {reference_country} on {topic_group}',
            template="plotly_white",
            height=fig_height,
        )
        fig.update_traces(mode='lines+markers', line_shape='spline')
        fig.update_layout(yaxis_range=[0,1], font=dict(family="Helvetica"))
        
        mean_similarity = filtered_df.groupby('Year')['Similarity'].mean()
        fig.add_scatter(x=mean_similarity.index, 
        y=mean_similarity.values, mode='lines', 
        name='Average', 
        line=dict(color='black', shape='spline', width=5))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

    