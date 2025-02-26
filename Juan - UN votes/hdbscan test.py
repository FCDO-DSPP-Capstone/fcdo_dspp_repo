

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
