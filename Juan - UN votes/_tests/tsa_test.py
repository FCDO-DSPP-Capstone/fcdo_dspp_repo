# TSA test

####################################
## 0. PACKAGES ##################
####################################

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
import janitor
from itertools import combinations
from numpy.linalg import LinAlgError

####################################
## 1. IMPORT DATA ##################
####################################

un_data = pd.read_csv('~/Library/CloudStorage/Dropbox/5. LSE MPA DSPP/Cursos/Capstone PP4B5/fcdo_dspp_capstone/un_data/2024_09_12_ga_resolutions_voting.csv')

# online location
# un_data = pd.read_csv('https://digitallibrary.un.org/record/4060887/files/2024_09_12_ga_resolutions_voting.csv?')

# cleaning country names to have a 1:1 match to country code
df_names = un_data.drop_duplicates(subset=['ms_code'], keep='first')
df_names = df_names[['ms_code', 'ms_name']]

un_data = un_data.drop(columns='ms_name')  # Drop the original ms_name column to avoid conflicts
un_data = pd.merge(un_data, df_names, on='ms_code', how='left')  # re-merge the country name column

####################################
## 2. Rolling TSA with Clustering ##
####################################

# selecting vars
un_votes = un_data[['undl_id', 'ms_code', 'ms_name', 'ms_vote', 'date']]

# Recode votes
recode_map = {'A': 0, 'Y': 1, 'N': -1}
un_votes['ms_vote'] = un_votes['ms_vote'].map(recode_map)
un_votes['year'] = pd.to_datetime(un_votes['date']).dt.year  # Extract year
un_votes['undl_id'] = un_votes['undl_id'].astype(str)

# Sort data by year
un_votes = un_votes.sort_values('year')

# Initialize an empty DataFrame to store TSA and cluster results
results = pd.DataFrame(columns=['ms_code', 'ms_name', 'Factor1', 'Factor2', 'year'])

# Rolling TSA and clustering process
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

    try:
        # Perform TSA using TSNE
        tsa_result = TSNE(n_components=2, random_state=42).fit_transform(scaled_data)
    except LinAlgError:
        tsa_result = np.full((scaled_data.shape[0], 2), np.nan)

    #Rotate to stabilize orientation with USA as anchor
    us_index = pivot_window[pivot_window['ms_code'] == 'USA'].index
    if len(us_index) > 0:
        us_x, us_y = tsa_result[us_index[0]]

        # Determine transformation needed
        flip_x = -1 if us_x > 0 else 1
        flip_y = -1 if us_y > 0 else 1

        # Apply transformation
        tsa_result[:, 0] *= flip_x
        tsa_result[:, 1] *= flip_y

    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels = clustering.fit_predict(np.nan_to_num(tsa_result))
    labels = labels.astype(str)

    # Combine TSA results and clustering labels with metadata
    window_results = pd.DataFrame({
        'ms_code': pivot_window['ms_code'],
        'ms_name': pivot_window['ms_name'],
        'Factor1': tsa_result[:, 0] if tsa_result is not None else np.nan,
        'Factor2': tsa_result[:, 1] if tsa_result is not None else np.nan,
        'year': current_year,
        'cluster': labels
    })

    # Append to the results DataFrame
    results = pd.concat([results, window_results], ignore_index=True)

results['cluster'] = results['cluster'].astype('int').astype('category')


import plotly.express as px

# Creating an interactive scatter plot with hover functionality
fig_h = px.scatter(
    results,
    x='Factor1',
    y='Factor2',
    color='cluster',
    color_discrete_map= {
    "-1": 'black',
    "0": 'blue',
    "1": 'orange',
    "2": 'green',
    "3": 'grey'},
    animation_frame='year',
    hover_data={'ms_code': True, 'ms_name': True, 'year': True, 'cluster': True,
    'Factor1': False, 'Factor2': False},
    title="PCA Clustering with HDBSCAN",
    labels={"Factor1": "Principal Component 1", "Factor2": "Principal Component 2"}
)

# Displaying the plot
fig_h.show()


