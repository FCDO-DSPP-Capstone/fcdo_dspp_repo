import requests
import pandas as pd
def fetch_openalex_data(concept_id, topic_label, start_year, end_year):
    url = "https://api.openalex.org/works"
    params = {
        'filter': f'concepts.id:{concept_id},publication_year:{start_year}-{end_year},type:article',
        'per-page': 200,
        'mailto': 'juani.piquer@gmail.com'
    }
    
    extracted_data = []
    cursor = "*"  #openalex api docuemntation says this will start pagination

    while cursor:
        params['cursor'] = cursor
        response = requests.get(url, params=params)
        data = response.json()

        works = data.get('results', [])
        for work in works:
            year = work.get('publication_year', None)
            title = work.get('title', 'Unknown Title')
            authors = ', '.join(
                [author['author']['display_name'] for author in work.get('authorships', []) if author['author'].get('display_name')]
            )

            institutions = [auth.get('institutions', []) for auth in work.get('authorships', [])]
            
            #institution names and countries
            institution_names = set()
            institution_countries = set()
            for inst_group in institutions:
                for inst in inst_group:
                    if inst.get('display_name'):  #avoid NoneType errors
                        institution_names.add(inst.get('display_name', 'Unknown Institution'))
                    if inst.get('country_code'):  #again avoid NoneType errors
                        institution_countries.add(inst.get('country_code', 'Unknown Country'))
            
            citation_count = work.get('cited_by_count', 0)

            extracted_data.append({
                'Year': year,
                'Title': title,
                'Authors': authors,
                'Institution Name': ', '.join(institution_names) if institution_names else 'Unknown Institution',
                'Institution Country': ', '.join(institution_countries) if institution_countries else 'Unknown Country',
                'Citation Count': citation_count,
                'Topic': topic_label
            })
        
        cursor = data.get('meta', {}).get('next_cursor') 

    return extracted_data

#each topic's data
start_year = 2010
end_year = 2025

ai_data = fetch_openalex_data('C154945302', 'Artificial Intelligence', start_year=start_year, end_year=end_year)
qt_data = fetch_openalex_data('C190463098', 'Quantum Technology', start_year=start_year, end_year=end_year)
eb_data = fetch_openalex_data('C136229726', 'Engineering Biology', start_year=start_year, end_year=end_year)

#combine
all_data = ai_data + qt_data + eb_data

#convert to pandas df
df = pd.DataFrame(all_data)

#save as CSV
df.to_csv("Grant - OpenAlex/openalex_combined_dataset.csv", index=False)

print("Dataset saved as 'openalex_combined_dataset.csv'")

