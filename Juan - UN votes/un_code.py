### UN voting process source code ### 
""" this script firstly, import the UN general assembly voting data, obtained from 
the UN digital library. The dataset consisstes of all votes on the general assembly from 1948 to
september 2024.

Then it performs one PCA with n_components=2 per year over a window of Five years of votes.

Then, performs a clustering based on the PBA with n clusters = 4 

Then it calculate a pair wise Co-cluster score over time

Then it calcualtes a country Pair wise mean distance of their PCA's
"""
####################################
## 1. PACKAGES ##################
####################################

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import plotly.express as px



####################################
## 1. IMPORT DATA ##################
####################################
un_data = pd.read_csv('~/Library/CloudStorage/Dropbox/5. LSE MPA DSPP/Cursos/Capstone PP4B5/fcdo_dspp_capstone/un_data/2024_09_12_ga_resolutions_voting.csv')

#online location 
#un_data = pd.read_csv('https://digitallibrary.un.org/record/4060887/files/2024_09_12_ga_resolutions_voting.csv?')

# cleaning country names to have a 1:1 match to country code

df_names = un_data.drop_duplicates(subset=['ms_code'], keep='first')
df_names = df_names[['ms_code', 'ms_name']]

un_data = un_data.drop(columns='ms_name')  # Drop the original ms_name column to avoid conflicts
un_data = pd.merge(un_data, df_names, on='ms_code', how='left')  # re merge the country name column


####################################
## 2. Rolling PCA with Clustering ##
####################################
""" 
1. Calculate PCA for a rolling window of 5 years. Each year has its own PCA.
2. Calcualte clusters for EACH of the PCAs
3. Plotting
"""

# selecting vars 
un_votes = un_data[['undl_id', 'ms_code', 'ms_name', 'ms_vote', 'date']]

# Recode votes
recode_map = {'A': 0, 'Y': 1, 'N': -1}
un_votes['ms_vote'] = un_votes['ms_vote'].map(recode_map)
un_votes['year'] = pd.to_datetime(un_votes['date']).dt.year  # Extract year
un_votes['undl_id'] = un_votes['undl_id'].astype(str)

# Sort data by year
un_votes = un_votes.sort_values('year')

# Initialize an empty DataFrame to store PCA and cluster results
results = pd.DataFrame(columns=['ms_code', 'ms_name', 'PCA1', 'PCA2', 'year'])

# Rolling PCA and clustering process
years = sorted(un_votes['year'].unique())
window_size = 5

for idx, current_year in enumerate(years):
    # Determine the range of years to include in the rolling window
    start_idx = max(0, idx - window_size + 1)
    years_window = years[start_idx:idx + 1]
    
    # Filter data for the rolling window
    window_data = un_votes[un_votes['year'].isin(years_window)]
    
    # Pivot data to wide format
    pivot_window = window_data.pivot(
        index=['ms_code', 'ms_name'], 
        columns='undl_id', 
        values='ms_vote'
    ).fillna(0).reset_index()

    # Extract numerical data and preserve column names
    numerical_data = pivot_window.loc[:, pivot_window.columns.difference(['ms_code', 'ms_name'])]
    
    # One-Hot Encode Numerical Columns
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoded = one_hot_encoder.fit_transform(numerical_data)

    # Combine one-hot encoded data with the rest of the DataFrame
    # Use the original column names to generate feature names
    names = one_hot_encoder.get_feature_names_out()
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=names)

# Concatenate with the rest of the DataFrame
    processed_data = pd.concat(
    [pivot_window.drop(columns=numerical_data).reset_index(drop=True), 
    one_hot_encoded_df],
    axis=1
    )
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_data.select_dtypes(include=[np.number]))
    
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    # Normalize the PCA output
    pca_result = scaler.fit_transform(pca_result)


    #Rotate to stabilize orientation with USA as anchor
    us_index = pivot_window[pivot_window['ms_code'] == 'USA'].index
    if len(us_index) > 0:
        us_x, us_y = pca_result[us_index[0]]

        # Determine transformation needed
        flip_x = -1 if us_x > 0 else 1
        flip_y = -1 if us_y > 0 else 1

        # Apply transformation
        pca_result[:, 0] *= flip_x
        pca_result[:, 1] *= flip_y

    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels = clustering.fit_predict(pca_result)
    labels = labels.astype(str)


    # Combine PCA results and clustering labels with metadata
    window_results = pd.DataFrame({
        'ms_code': pivot_window['ms_code'],
        'ms_name': pivot_window['ms_name'],
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'year': current_year,
        'cluster': labels
    })

    # Append to the results DataFrame
    results = pd.concat([results, window_results], ignore_index=True)


results['cluster'] = results['cluster'].astype('int').astype('category')



#######################################################
## 3. Table of Co-Cluster score with country filter ##
#######################################################

# Create an interactive scatter plot with a slider
fig = px.scatter(
    results,
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

fig_norm = fig
fig_norm.show()

#### tables of more and least clustered 
import pandas as pd
import numpy as np
from itertools import combinations


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