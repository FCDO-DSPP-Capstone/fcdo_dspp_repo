# FCDO data science capstone 


Web page: https://fcdo-dspp-capstone.github.io/fcdo_dspp_repo/


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
3. OpenAlex: [Networks](https://networkx.org)  
4. Comtrade: Networks  

----------------------------
This project was developed as the 2025 Capstone project for the MPA - Data Science for Public Policy program at the London School of Economics.  
Authors: Grant Benton, Juan Piquer, Cat Russell and Rhea Soni
  
Thanks to our LSE supervisor Alexander Evans, our FCDO client James Sherbrock, and our Data science tutor Casey Kearney.


----------------------------


Suggested next steps in development;
    