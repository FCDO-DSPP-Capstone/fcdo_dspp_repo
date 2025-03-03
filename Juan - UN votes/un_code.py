### UN voting process source code ### 
""" This scripts imports the results from the PCA and clustering of UN votes to 
make an app with plots, table and more information
It also enrich the data set with world bank data for population and GDP

"""
####################################
## 0. PACKAGES ##################
####################################

import pandas as pd
import numpy as np
import plotly.express as px
from itertools import combinations

####################################
## 1. IMPORT DATA ##################
####################################
pca_results = pd.read_csv('Juan - UN votes/pca_results.csv')

pca_results['cluster'] = pca_results['cluster'].astype('category')

#######################################################
## 2. UN PCA votes plot with year slider ##
#######################################################

fig = px.scatter(
    pca_results,
    x='PCA1',
    y='PCA2',
    color='cluster',
    animation_frame='year',
    hover_data={'ms_name': True, 'PCA1': False, 'PCA2': False},
    title='PCA on five year rolling window',
    labels={'PCA1': 'Principal component 1', 'PCA2': 'Principal component 2'},
    template='plotly_white'
    
)

fig.update_layout(
    font=dict(
        family='Helvetica', 
        size=14,         # Font size
        color='black'     # Font color
    ),
    title_x=0.5  # Center the title
)

# Show the plot
fig.show()

fig.write_html("outputs/interactive_pca_plot with windows.html", auto_play=False)




#######################################################
## 3. Table of Co-Cluster score with country filter ##
#######################################################

#### tables of more and least clustered 



# Compute Co-Clustering Score
countries = results['ms_name'].unique() 
co_cluster_matrix = pd.DataFrame(0, index=countries, columns=countries)

for year in results[results['year'] >= 1990]['year'].unique():
    year_data = results[results['year'] == year]
    clusters = year_data.groupby('cluster')['ms_name'].apply(list)
    
    for country_list in clusters:
        for c1, c2 in combinations(country_list, 2):
            co_cluster_matrix.loc[c1, c2] += 1
            co_cluster_matrix.loc[c2, c1] += 1

co_cluster_matrix = co_cluster_matrix / results[results['year'] >= 1990]['year'].nunique()

# Top 10 most clustered together pairs
top_10_pairs = co_cluster_matrix.unstack().sort_values(ascending=False)
top_10_pairs_table = pd.DataFrame(top_10_pairs, columns=['Co-Cluster Score']).reset_index()
top_10_pairs_table.columns = ['Country 1', 'Country 2', 'Co-Cluster Score']

print("Top 10 Most Clustered Together Pairs:")
print(top_10_pairs_table)


import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
from dash.dash_table import DataTable, FormatTemplate


# Initialize Dash app
app = dash.Dash(__name__)

percentage = FormatTemplate.percentage(2)

app.layout = html.Div([
    html.H3("Country Pairwise Co-Cluster Table"),
    
    # Dropdown to select Country 1
    dcc.Dropdown(
        id='country-1-dropdown',
        options=[{'label': c, 'value': c} for c in sorted(top_10_pairs_table['Country 1'].unique())],
        value=top_10_pairs_table['Country 1'].unique()[0],  # Default selection
        clearable=False
    ),
    
    # Sorting Button
    html.Button("Sort: Ascending", id="sort-button", n_clicks=0),

    # DataTable to display filtered data
    dash_table.DataTable(
        id='filtered-table',
        columns=[
            {'name': 'Country 2', 'id': 'Country 2'},
            dict(name='Co-Cluster Score', id='Co-Cluster Score', type='numeric', format=percentage)
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        sort_action="none"  # We handle sorting manually
    )
])

@app.callback(
    Output('filtered-table', 'data'),
    Output('sort-button', 'children'),
    Input('country-1-dropdown', 'value'),
    Input('sort-button', 'n_clicks')
)
def update_table(selected_country, n_clicks):
    # Filter based on selected "Country 1"
    filtered_df = top_10_pairs_table[top_10_pairs_table['Country 1'] == selected_country].copy()
    
    # Determine sorting order
    ascending = n_clicks % 2 == 0  # Toggle sorting direction
    filtered_df = filtered_df.sort_values('Co-Cluster Score', ascending=ascending)
    
    # Update button text
    button_text = "Sort: Ascending" if ascending else "Sort: Descending"

    return filtered_df.to_dict('records'), button_text

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


#### compute average distance between countries in PCA space 
## Compute alongside co clusters. Allows to see the Partner grouping rate as well as a more indivudal distance rate.

from sklearn.metrics import pairwise_distances

# Assume 'results' contains PCA1, PCA2, ms_name (Country), and year
pca_results = results[results['year'] >= 1994].copy()

def compute_pairwise_distances(pca_results):
    # Store distance results
    all_distances = []

    # Compute pairwise distances per year
    for year in pca_results['year'].unique():
        yearly_data = pca_results[pca_results['year'] == year]
        
        # Extract country names and PCA coordinates
        country_names = yearly_data['ms_name'].values
        pca_coords = yearly_data[['PCA1', 'PCA2']].values

        # Compute pairwise Euclidean distances
        distances = pairwise_distances(pca_coords, metric='euclidean')
        
        # Convert to DataFrame in long format
        distance_df = pd.DataFrame(distances, index=country_names, columns=country_names).reset_index()
        distance_df = pd.DataFrame({'Country 1': np.repeat(country_names, len(country_names)), 
                                    'Country 2': np.tile(country_names, len(country_names)), 
                                    'distance': distances.flatten()})
        distance_df['year'] = year
        
        all_distances.append(distance_df)

    # Concatenate all yearly results
    long_format_distances = pd.concat(all_distances, ignore_index=True)

    # Remove self-distances (Country 1 == Country 2)
    long_format_distances = long_format_distances[long_format_distances['Country 1'] != long_format_distances['Country 2']]

    # Compute the mean distance per (Country 1, Country 2) pair
    mean_distance_per_pair = long_format_distances.groupby(['Country 1', 'Country 2'])['distance'].mean().reset_index()
    mean_distance_per_pair = mean_distance_per_pair.sort_values('distance', ascending=False)

    return mean_distance_per_pair

# Compute pairwise distances
mean_distance_per_pair = compute_pairwise_distances(pca_results)

# Display result
print(mean_distance_per_pair)



### Compute the total mean score of entropy of each country
# this should be added as "Country Entropy: XX" as text beofre the table 
# but after the Country selector. 
# Calcualteit 


### Separate votes by country 
## Agregarle botón para seleccionar que países resaltar desde la lista completa para any givren year
# eso hay q hacerlo en dash



## Agregar World bank data. 
