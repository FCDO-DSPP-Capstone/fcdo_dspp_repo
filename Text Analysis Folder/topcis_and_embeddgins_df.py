
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