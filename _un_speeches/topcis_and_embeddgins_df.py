
import pandas as pd
import numpy as np
import os
import re
import bertopic
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
import pycountry
from sentence_transformers import SentenceTransformer

## 1. Creating speeches CSV
## Each speech is splitted into Sentences. Each sentence is processed then as a separate object,
## with a country and year label, and a topic labels to be asigned by Transformer models.



root_folder_path = 'Text Analysis Folder/UN_Speeches'

def parse_filename(filename):
    name_parts = filename.replace('.txt', '').split('_')

    if len(name_parts) < 3:
        return None, None, None  

    country = name_parts[0]
    session = name_parts[1]
    
    try:
        year = int(name_parts[2])
    except ValueError:
        return None, None, None  

    return country, session, year

def preprocess_text(text):
    # Temporarily replace dots in common abbreviations with placeholders
    abbreviation_patterns = [r'\bMs\.', r'\bMr\.', r'\bDr\.', r'\bPresident\.']
    for pattern in abbreviation_patterns:
        text = re.sub(pattern, lambda match: match.group(0).replace('.', '[DOT]'), text)
    return text

def postprocess_sentence(sentence):
    return sentence.replace('[DOT]', '.')

sentence_data = []

for subfolder in os.listdir(root_folder_path):
    subfolder_path = os.path.join(root_folder_path, subfolder)
    
    if os.path.isdir(subfolder_path) and "Session" in subfolder:
        for filename in os.listdir(subfolder_path):
            if not filename.endswith('.txt') or filename.startswith('.'):  
                continue  # Skip hidden/system files like .DS_Store
            
            country, session, year = parse_filename(filename)
            if country is None:  # Skip if parsing failed
                continue  
            
            file_path = os.path.join(subfolder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
            
            # Preprocess text
            text_content = preprocess_text(text_content)
            text_content = text_content.replace('\n', ' ')  # Replace line breaks with spaces
            
            # Split by full stops while ignoring placeholders for abbreviations
            sentences = re.split(r'(?<!\[DOT])\.\s+', text_content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence: 
                    sentence_data.append([country, session, year, postprocess_sentence(sentence)])

sentence_df = pd.DataFrame(sentence_data, columns=['Country', 'Session', 'Year', 'Sentence'])

country_code_to_name = {country.alpha_3: country.name for country in pycountry.countries}
sentence_df['Country Name'] = sentence_df['Country'].map(country_code_to_name)


#############################################################################
## 2. Creatin topics using BERTopic model and embeddgins Sentence transformer
#############################################################################

zeroshot_topic_dict = {
    # Military technology
    0: "Nuclear Weapons",
    1: "Biological Weapons",
    2: "Chemical Weapons",
    # Dual Use Technology
    3: "Artificial Intelligence",
    4: "Quantum Technology",
    5: "Supply Chains",
    6: "Data Security",
    7: "Biological Engineering",
    8: "General Emerging Technologies",
    9: "Nuclear Energy",
    10: "Energy Security",
    # Civilian technology 
    11: "Climate Change and Renewable Energy",
    12: "Electric Cars",
    13: "Financial Technology",
    14: "Health Technology",
    15: "Technology in Education",
    # Capacity Building 
    16: "Digital Divide",
    17: "Technology Capacity Building",
    18: "Technology Transfer",
    # Grey Zone
    19: "Cyberattacks and Offensive Cyber",
    20: "Election Interference",
    21: "Cyber Espionage"
}


## Create embeddgins using sentence transformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Pass the sentences to a list to be feeded on Sentence tranformer
sentence_docs = sentence_df["Sentence"].tolist()
# Generate embeddings for the sentences
sentence_embeddings = embedding_model.encode(sentence_docs, show_progress_bar=True)

### Fit BERTopic model 
# define zero shot BERTopic model 
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2", 
    min_topic_size=15,
    zeroshot_topic_list=list(zeroshot_topic_dict.values()),
    zeroshot_min_similarity=.5,
    representation_model=KeyBERTInspired()
)

# fit the model to the sentence dataframe
topics, _ = topic_model.fit_transform(sentence_docs, sentence_embeddings)

# add the topic label created to the full dataframe 
sentence_df['Topic'] = topics


# Map the numeric topics to their descriptive labels in the dataframe
sentence_df["Topic Name"] = sentence_df["Topic"].map(zeroshot_topic_dict)

# Filter data to left only labeled sentences 
tech_topcis_df = sentence_df[sentence_df["Topic Name"].notna()].copy()

# Filter embeddings to left only the ones from labeled sentences
tech_embeddings = sentence_embeddings[tech_topcis_df.index]

# clean up environment
del sentence_df, embedding_model, topic_model, sentence_docs, sentence_embeddings, topics, zeroshot_topic_dict, _


#############################################################################
############# 3.  export filtered embeddings and data frame 
#############################################################################

np.save('Text Analysis Folder/data_out/tech_embeddings.npy', tech_embeddings)

tech_topcis_df.to_csv('Text Analysis Folder/data_out/tech_topcis_df.csv', index=False)

#############################################################################
############# 4. Re import and generation of cosine simiality data frame 
#############################################################################

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


sentence_df = pd.read_csv("Text Analysis Folder/data_out/tech_topcis_df.csv")
embeddings = np.load("Text Analysis Folder/data_out/tech_embeddings.npy")

topic_groups = {
    "Military Technology": [0, 1, 2],  # Nuclear Weapons, Biological Weapons, Chemical Weapons
    "Dual Use Technology": [3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 12, 13, 14, 15, 16, 17, 18],  # AI, Quantum Tech, etc.
    "Civilian Technology": [11, 12, 13, 14, 15, 16, 17, 18],  # Climate change, Electric cars, etc.
}

country_groups = {
    "ASEAN": ["Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam", "India", "Pakistan"],
    "European Union": ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"],
    "African Union": ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo", "Democratic Republic of the Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"],
    "Middle East": ["Afghanistan", "Bahrain", "Cyprus", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen"],
    "Latin America and Caribbean": ["Antigua and Barbuda", "Argentina", "Bahamas", "Barbados", "Belize", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "Ecuador", "El Salvador", "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago", "Uruguay", "Venezuela"],
    "Reference": ["United States", "United Kingdom", "China"]
}

# Add geographic group column
def assign_country_group(country):
    for group, countries in country_groups.items():
        if country in countries:
            return group
    return None

sentence_df['Country Group'] = sentence_df['Country Name'].apply(assign_country_group)

# Add macro topic column
def assign_topic_group(topic_id):
    for group_name, topic_ids in topic_groups.items():
        if topic_id in topic_ids:
            return group_name
    return None

sentence_df['Macro Topic'] = sentence_df['Topic'].apply(assign_topic_group)

# Ensure 'Year' is an integer
sentence_df['Year'] = sentence_df['Year'].astype(int)

# Group by rows and compute average embeddings
def average_embeddings_five_year_window(df, embeddings, group_vars):
    # Add an index column to the dataframe to keep track of original indices
    df['index'] = np.arange(len(df))
    
    # Initialize lists to store the grouped keys and average embeddings
    grouped_keys = []
    avg_embeddings_list = []
    
    # Get the unique years in the dataframe
    unique_years = df['Year'].unique()
    
    # Iterate over each unique combination of ['Country Name', 'Country Group', 'Macro Topic']
    for name, group in df.groupby(group_vars[:-1]):
        for year in unique_years:
            # Define the five-year window
            window = range(year - 3, year + 4)
            
            # Get the indices of the rows within the five-year window
            window_indices = group[group['Year'].isin(window)].index
            
            if len(window_indices) > 0:
                # Compute the average embedding for the current group and window
                avg_embedding = embeddings[window_indices].mean(axis=0)
                
                # Append the group keys and year to the grouped_keys list
                grouped_keys.append((*name, year))
                avg_embeddings_list.append(avg_embedding)
    
    # Convert the grouped_keys list to a dataframe
    grouped_keys_df = pd.DataFrame(grouped_keys, columns=group_vars)
    grouped_keys_df = grouped_keys_df.sort_values(by=['Year'])
    # Convert the avg_embeddings_list to a numpy array
    avg_embeddings_array = np.vstack(avg_embeddings_list)
    
    return grouped_keys_df, avg_embeddings_array

group_vars = ['Country Name', 'Country Group', 'Macro Topic', 'Year']

grouped_keys, avg_embeddings = average_embeddings_five_year_window(sentence_df, embeddings, group_vars)

print(grouped_keys)
print(avg_embeddings)




def calculate_cosine_similarity(grouped_keys, avg_embeddings):
    # Initialize an empty list to store the results
    results = []

    # Get the unique macro topics and reference countries
    macro_topics = grouped_keys['Macro Topic'].unique()
    reference_countries = grouped_keys[grouped_keys['Country Group'] == 'Reference']['Country Name'].unique()

    # Iterate over each macro topic
    for topic in macro_topics:
        # Iterate over each reference country
        for ref_country in reference_countries:
            # Get the reference embedding for the selected reference country, macro topic, and year 2021
            reference_embedding = avg_embeddings[(grouped_keys['Country Name'] == ref_country) & 
                                                 (grouped_keys['Macro Topic'] == topic) & 
                                                 (grouped_keys['Year'] == 2021)][0]
            
            # Filter the DataFrame based on the macro topic
            topic_df = grouped_keys[grouped_keys['Macro Topic'] == topic]
            
            # Iterate over each country in the topic DataFrame
            for country in topic_df['Country Name'].unique():
                country_df = topic_df[topic_df['Country Name'] == country]
                
                # Iterate over each year for the current country
                for year in country_df['Year'].unique():
                    country_embedding = avg_embeddings[(grouped_keys['Country Name'] == country) & 
                                                       (grouped_keys['Macro Topic'] == topic) & 
                                                       (grouped_keys['Year'] == year)][0]
                    similarity = cosine_similarity([reference_embedding], [country_embedding])[0][0]
                    
                    # Append the result to the list
                    results.append({
                        'Country Name': country,
                        'Country Group': country_df['Country Group'].iloc[0],
                        'Macro Topic': topic,
                        'Year': year,
                        'Reference': ref_country,
                        'Similarity': similarity
                    })
    
    # Create a DataFrame from the results
    similarity_df = pd.DataFrame(results)
    return similarity_df.sort_values(by=['Country Name', 'Macro Topic', 'Year'])

similarity_df = calculate_cosine_similarity(grouped_keys, avg_embeddings)



export = similarity_df.to_csv('Text Analysis Folder/data_out/similarity_df.csv', index=False)