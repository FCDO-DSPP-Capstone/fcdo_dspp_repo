### UN voting process source code ### 
""" this script firstly, import the UN general assembly voting data, obtained from 
the UN digital library. The dataset consisstes of all votes on the general assembly from 1948 to
september 2024.

Then it performs one PCA with n_components=2 per year over a window of Five years of votes.

Then, performs a aglomerative clustering over the PCA results with n clusters = 4

Optimal numer of clusters and PC components calulated with elbow methods in older code.

Finally, merge the results with World Bank data on population, gdp and gdp per capita
World bank data transformed to percentiles with 100 quantiles rather than absolute values.

Export is a long dataframe with country, country code, pca1, pca2, year, cluster and world bank dataa
"""

####################################
## 0. PACKAGES ##################
####################################

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer, MinMaxScaler
from sklearn.metrics import pairwise_distances
import janitor
from itertools import combinations


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

    # Apply arcsinh transformation to complress outliers 
    window_results['PCA1'] = np.arcsinh(window_results['PCA1'])
    window_results['PCA2'] = np.arcsinh(window_results['PCA2'])

    

    # Append to the results DataFrame
    results = pd.concat([results, window_results], ignore_index=True)


results['cluster'] = results['cluster'].astype('int').astype('category')


####################################
## 3. Merge with world bank data  ##
####################################

# Import and clean world bank data (Population, GDP total and GDP PP.)
wb_data = (pd.read_csv(
    'Juan - UN votes/world_bank_data/world_bank_data.csv')
    .clean_names().dropna(subset=['series_code']))

# get the least of years
year_columns = wb_data.filter(regex=r'^\d{4}_\[yr\d{4}\]$').columns

# replace all ".." values in year columns by NA
wb_data[year_columns] = wb_data[year_columns].apply(
    lambda x: pd.to_numeric(x, errors='coerce')
)

# Fill the "latest_value" var with the latest value for each series 
wb_data["latest_value"] = wb_data[year_columns].ffill(axis=1).iloc[:, -1]

# Replace 'series_code' with shorter names
series_code_map = {
    'SP.POP.TOTL': 'pop',
    'NY.GDP.MKTP.CD': 'gdp_tot',
    'NY.GDP.PCAP.CD': 'gdp_pp'
}
wb_data['series_code'] = wb_data['series_code'].replace(series_code_map)

# Pivot the wb_data to wide, leaving series code as col names and using only the last value for each series
wb_data_wide = wb_data.pivot(
    index=['country_name', 'country_code'],
    columns='series_code',
    values='latest_value'
).reset_index()


### country groups labels
region_data = (pd.read_csv(
    'Juan - UN votes/world_bank_data/country_groups_data_un.csv',
    sep=';'
)
    .clean_names()
)

# Select relevant columns for region name and ISO alpha-3 code
region_data = region_data[['region_name', 'iso_alpha3_code']]
# merge un regions with wb data
wb_data_wide = pd.merge(
    wb_data_wide,
    region_data,
    left_on='country_code',
    right_on='iso_alpha3_code',
    how='left'
).drop('iso_alpha3_code', axis=1)


# Merge pca results with world bank data
pca_results = pd.merge(
    results, wb_data_wide, 
    left_on=["ms_code"], 
    right_on=["country_code"], 
    how="left"
)

pca_results = pca_results.drop(columns=['country_code', 'country_name'])


# Normalize GDP columns to be between 0 and 1 and quantize by percentiles
scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform')

# Apply scaling to GDP columns
pca_results[['gdp_tot', 'gdp_pp', 'pop']] = scaler.fit_transform(pca_results[['gdp_tot', 'gdp_pp', 'pop']])


#######################################################
## 4. Co cluster scores and pairwise distance score  ##
#######################################################

#### tables of more and least clustered 


# Compute Co-Clustering Score
countries = pca_results[pca_results['year'] >= 1990]['ms_name'].unique()
co_cluster_matrix = pd.DataFrame(0, index=countries, columns=countries)

for year in pca_results[pca_results['year'] >= 1990]['year'].unique():
    year_data = pca_results[pca_results['year'] == year]
    clusters = year_data.groupby('cluster')['ms_name'].apply(list)
    
    for country_list in clusters:
        for c1, c2 in combinations(country_list, 2):
            co_cluster_matrix.loc[c1, c2] += 1
            co_cluster_matrix.loc[c2, c1] += 1

co_cluster_matrix = co_cluster_matrix / pca_results[pca_results['year'] >= 1990]['year'].nunique()

# Top 10 most clustered together pairs
co_cluster_table = co_cluster_matrix.unstack().sort_values(ascending=False)
co_cluster_table = pd.DataFrame(co_cluster_table, columns=['Co-Cluster Score']).reset_index()
co_cluster_table.columns = ['Country 1', 'Country 2', 'Co-Cluster Score']

co_cluster_table['Country 1'] = co_cluster_table['Country 1'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
co_cluster_table['Country 2'] = co_cluster_table['Country 2'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))

#### compute average distance between countries in PCA space 
## Compute alongside co clusters. Allows to see the Partner grouping rate as well as a more indivudal distance rate.


# Assume 'results' contains PCA1, PCA2, ms_name (Country), and year
pca_results_filter = pca_results[pca_results['year'] >= 1990].copy()

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
mean_distance_per_pair = compute_pairwise_distances(pca_results_filter)


# Capitalize the first letter of each word in 'Country 1' and 'Country 2' columns
mean_distance_per_pair['Country 1'] = mean_distance_per_pair['Country 1'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
mean_distance_per_pair['Country 2'] = mean_distance_per_pair['Country 2'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))

# Normalize distance values to be numbers from 0 to 1
scaler = MinMaxScaler()
mean_distance_per_pair['distance'] = scaler.fit_transform(mean_distance_per_pair[['distance']])

mean_distance_per_pair = mean_distance_per_pair.rename(columns={'distance': 'Mean distance'})

# Merge mean_distance_per_pair and co_cluster_table by 'Country 1' and 'Country 2'
association_scores = pd.merge(
    mean_distance_per_pair,
    co_cluster_table,
    on=['Country 1', 'Country 2'],
    how='right'
)


####################################
## 5. export pca_results      ##
####################################

pca_results.to_csv('Juan - UN votes/pca_results.csv', index=False)

association_scores.to_csv('Juan - UN votes/association_scores.csv', index=False)