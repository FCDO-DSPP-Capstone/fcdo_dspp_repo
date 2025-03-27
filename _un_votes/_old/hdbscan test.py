

import hdbscan
def hdbscan_clustering(year, min_cluster_size, min_samples):

    pca_result = results[results['year'] == year].copy()
    pca_for_clustering = pca_result[['PCA1', 'PCA2']]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=-1, min_samples=min_samples)

    clusterer.fit(pca_for_clustering)

    clusterer.labels_

    labels_h = pd.Categorical(clusterer.labels_)

    hd_clust = pd.DataFrame({

            'ms_code': pca_result['ms_code'],
            'ms_name': pca_result['ms_name'],
            'PCA1': pca_for_clustering['PCA1'],
            'PCA2': pca_for_clustering['PCA2'],
            'year': year,
            'cluster': labels_h
        })

    # Creating an interactive scatter plot with hover functionality
    fig_h = px.scatter(
        hd_clust,
        x='PCA1',
        y='PCA2',
        color='cluster',
        color_discrete_map= {
        -1 : 'black'},
        hover_data={'ms_code': True, 'ms_name': True, 'year': True, 'cluster': True,
        'PCA1': False, 'PCA2': False},
        title="PCA Clustering with HDBSCAN",
        labels={"PCA1": "Principal Component 1", "PCA2": "Principal Component 2"}
    )

    # Displaying the plot
    fig_h.show()

year = 2005
min_cluster_size = 10
min_samples = 5

hdbscan_clustering(year, min_cluster_size, min_samples)




## GMM cluster test and HDBSCAN with probabilities
#######################################################
## 2.1 GMM over PCA results with year slider ##
#######################################################

from sklearn.mixture import GaussianMixture
import hdbscan

# Fit GMM to each year
gmm_results = []
for year in pca_results['year'].unique():
    year_data = pca_results[pca_results['year'] == year][['PCA1', 'PCA2']]
    gmm = GaussianMixture(n_components=4)
    gmm.fit(year_data)
    gmm_results.append(pd.DataFrame(gmm.predict_proba(year_data), columns=[f'prob_{i}' for i in range(4)]))
gmm_results = pd.concat(gmm_results, axis=0, ignore_index=True)
# Plot results for year == 2024
cluster_assignments = gmm_results.loc[pca_results['year'] == 2024].idxmax(axis=1)
cluster_probabilities = gmm_results.loc[pca_results['year'] == 2024].max(axis=1)

fig_gmm = px.scatter(
    pca_results[pca_results['year'] == 2024],
    x='PCA1',
    y='PCA2',
    color=cluster_assignments,
    opacity=cluster_probabilities,
    hover_data={'ms_name': True, 'PCA1': False, 'PCA2': False},
    title='Gaussian Mixture Model on PCA results',
    labels={'color': 'Cluster'}
    
)

fig_gmm.update_layout(
    font=dict(
        family='Helvetica', 
        size=14,         # Font size
        color='black'     # Font color
    ),
    title_x=0.5  # Center the title
)

# Show the plot
fig_gmm.show()

# Fit HDBSCAN to each year
hdbscan_results = []
for year in pca_results['year'].unique():
    year_data = pca_results[pca_results['year'] == year][['PCA1', 'PCA2']]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, core_dist_n_jobs=-1)
    clusterer.fit(year_data)
    hdbscan_results.append(pd.DataFrame({'cluster': clusterer.labels_, 'probability': clusterer.probabilities_}, index=year_data.index))
hdbscan_results = pd.concat(hdbscan_results, axis=0, ignore_index=True)

hdbscan_results['cluster'] = hdbscan_results['cluster'].astype('category')

hdbscan_results.loc[hdbscan_results['cluster'] == -1, 'probability'] = 1

pca_hdbscan = pd.concat([pca_results.drop('cluster', axis=1), hdbscan_results], axis=1)

pca_hdbscan['opacity'] = pca_hdbscan['probability'].clip(lower=0, upper=1)
pca_hdbscan['opacity'] = pca_hdbscan['probability'].round(2)  # Rounding to 2 decimal places (if needed)

fig_hdbscan = px.scatter(
    pca_hdbscan[pca_hdbscan['year'] == 2024],
    x='PCA1',
    y='PCA2',
    color=pca_hdbscan[pca_hdbscan['year'] == 2024]['cluster'].astype('category'),
    opacity=pca_hdbscan[pca_hdbscan['year'] == 2024]['probability'],
    hover_data={'ms_name': True, 'PCA1': False, 'PCA2': False, 'probability': True},
    title='HDBSCAN on PCA results',
    labels={'color': 'Cluster'}
)

fig_hdbscan.update_layout(
    font=dict(
        family='Helvetica',
        size=14,
        color='black'
    ),
    title_x=0.5
)

fig_hdbscan.show()
