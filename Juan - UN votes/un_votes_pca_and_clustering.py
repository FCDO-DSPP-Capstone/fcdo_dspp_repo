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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
import janitor


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
    'Juan - UN votes/world_bank_data.csv')
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

####################################
## 4. export pca_results      ##
####################################

pca_results.to_csv('Juan - UN votes/pca_results.csv', index=False)

