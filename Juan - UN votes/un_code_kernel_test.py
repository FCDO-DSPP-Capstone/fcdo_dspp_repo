### UN voting process source code ### 
""" this script firstly, import the UN general assembly voting data, obtained from 
the UN digital library. The dataset consisstes of all votes on the general assembly from 1948 to
september 2024.

Then, it 



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
from sklearn.decomposition import KernelPCA



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
    pca = KernelPCA(n_components=2, kernel = "poly", n_jobs = -1)
    pca_result = pca.fit_transform(scaled_data)

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
## 3. Tables with scores of most clustered and stuff ##
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

