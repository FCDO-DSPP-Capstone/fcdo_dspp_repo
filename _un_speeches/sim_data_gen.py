
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
    html.H1("Country Speeches Similarity Analysis", style={'font-family': 'Helvetica'}),
    
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

    