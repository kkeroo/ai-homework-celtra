import pandas as pd
import wikipedia
import os
from tqdm import tqdm

def get_wikipedia_description(company_name):
    try:
        # Perform a Wikipedia search for the exact company name
        page_title = wikipedia.search(company_name, results=1)
        if page_title:
            summary = wikipedia.summary(page_title[0], sentences=20)  # Adjust sentences as needed
            return summary
        else:
            return "No Wikipedia page found"
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation by choosing first option
        try:
            summary = wikipedia.summary(e.options[0], sentences=2)
            return summary
        except:
            return "Disambiguation page encountered, description not retrieved"
    except Exception as e:
        return f"Error: {e}"

# File paths
input_file = 'resources/dataset.csv'
output_file = 'resources/dataset_with_descriptions.csv'

# Check if output file exists and load processed companies
processed_companies = set()
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    if 'DESCRIPTION' in existing_df.columns and 'COMPANY' in existing_df.columns:
        # Get companies that already have descriptions
        processed_companies = set(existing_df['COMPANY'])
    
# Load input data
df = pd.read_csv(input_file)

# Process companies and update CSV after each API call
for index, row in tqdm(df.iterrows(), total=len(df)):
    company_name = row['COMPANY']
    
    # Skip if already processed
    if company_name in processed_companies:
        continue
        
    # Get description
    description = get_wikipedia_description(company_name)
    
    # Create a new row with the description
    row_data = row.to_dict()
    row_data['DESCRIPTION'] = description
    
    # Append to the output file (create if doesn't exist)
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        pd.DataFrame([row_data]).to_csv(f, header=not os.path.exists(output_file) or os.path.getsize(output_file)==0, 
                                        index=False, sep=';')
    
    # Add to processed set
    processed_companies.add(company_name)

print(f"Processing complete. Results saved to {output_file}")