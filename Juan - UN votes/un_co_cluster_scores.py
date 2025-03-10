# Un co clusterings scores 


#### tables of more and least clustered 


# Compute Co-Clustering Score
countries = pca_results['ms_name'].unique() 
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
top_10_pairs = co_cluster_matrix.unstack().sort_values(ascending=False)
top_10_pairs_table = pd.DataFrame(top_10_pairs, columns=['Co-Cluster Score']).reset_index()
top_10_pairs_table.columns = ['Country 1', 'Country 2', 'Co-Cluster Score']

print("Top 10 Most Clustered Together Pairs:")
print(top_10_pairs_table)