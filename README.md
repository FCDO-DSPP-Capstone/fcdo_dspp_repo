# FCDO data science capstone 
Web page: https://fcdo-dspp-capstone.github.io/fcdo_dspp_repo/

----------------------------
This project was developed as the 2025 Capstone project for the MPA - Data Science for Public Policy program at the London School of Economics.  
Authors: Grant Benton, Juan Piquer, Cat Russell and Rhea Soni.

Thanks to our LSE supervisor Alexander Evans, our FCDO client James Sherbrock, and our Data science tutor Casey Kearney.
  
----------------------------

# Repo Structure

## `data_processing/`
Contains four folders, each dedicated to processing a specific dataset:
- **UN Votes**  
- **UN Speeches**  
- **Comtrade**  
- **OpenAlex (Scientific Research Papers)**  

## `web_page/apps/`
Contains four folders, each with the corresponding app for its respective dataset.
- The app scripts load the processed data, rearrange it for visualization, and render it using Dash.
- Each app is deployed on Posit Connect Cloud servers.

## `index.html`
- Serves as the main web page.
- Provides explanations and embeds the apps.
  
# Methods used
1. UN Votes: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  
2. UN Speeches: [BERTopic](https://maartengr.github.io/BERTopic/index.html)  
3. OpenAlex: [NetworkX](https://networkx.org)  
4. Comtrade: NetworkX 


# Suggested Next Steps in Development

## UN Votes
- **Filter Data:**  
  - Filter the dataset by topics to observe potential changes in distributions.  
  - Exclude Palestine-Israel topics for clearer analysis.  
- **Distance Analysis:**  
  - Measure the distance of each country from its continent's centroid.  
- **Continent Dispersion:**  
  - Calculate and track the change in continent dispersion over time (to see if regions become more or less concentrated).  
- **Feature Importance:**  
  - Identify the most significant votes or groups of votes using loadings extraction from PCA.

## UN Speeches
- **Topic Relevance:**  
  - Display the most relevant countries per topic. (Currently, only the most relevant topics per country are shown using pie charts.)  
- **Sentence Review:**  
  - Add a searchable table to view the underlying sentences emitted by a country with their assigned topic labels.  
- **Similarity Comparison:**  
  - Create a selector for “most similar sentences” to display pairs of sentences, allowing for direct comparisons of country speeches.  
- **Error Message:**  
  - In the similarity plot, display an "Empty" message when no data matches the selected filters.  
- **PCA Analysis:**  
  - Perform PCA on countries using the full sentence dataframe, with topics as features (`PCA (One row per country, one column per topic, values = count)`).  
  - Run a second PCA using only a subset of topics.  
  - Additionally, test averaging sentence embeddings by country and run PCA using the embedding dimensions as features.

## Scientific Research
- **Institution Analysis:**  
  - Add an explorer to identify the most relevant institutions per country and their strongest connections.  
- **Collaboration Networks:**  
  - Implement a filter to generate local collaboration networks by country.  
- **Research Insights:**  
  - Create a table or treemap displaying the most relevant papers in a given year or topic.

## Comtrade
- **Data Bug Fix:**  
  - Resolve the issue with underlying data for 2024, where some countries (e.g., China) show 0 exports incorrectly.  

