# FCDO data science capstone 


Web page: https://fcdo-dspp-capstone.github.io/fcdo_dspp_repo/


Repo structure:  
In data_processing there are four folers with the data process for each dataset: UN votes, UN Speeches, 
Comtrade and OpenAlex (scientific research papers).  

In web_page/apps there are four folder, each one with the correpsonding app for each data set.  
The apps code scripts loads the data generated in the data_processing scripts, rearrange it for plotting and render it.
Each apps is a Dash app deployed in Posit Connect Cloud servers.

Index html has the web page explanations text and the Apps embedded.
  
----------------------------
This project was developed as the 2025 Capstone project for the MPA - Data Science for Public Policy program at the London School of Economics.  
Authors: Grant Benton, Juan Piquer, Cat Russell and Rhea Soni
  
Thanks to our LSE supervisor Alexander Evans, our FCDO client James Sherbrock, and our Data science tutor Casey Kearney. 