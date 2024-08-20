import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT
import utils

system_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection, Entry Barrier and Accessibility, Market Instability, Illicit Activities, Influence of Large Financial Institutions and Crypto Institution
- Governance (G): Decentralized Governance Models (off-chain and on-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
"""

def create_prompt(title, content):
    return f"""
            Article Title: {title}
            Article Context: {content}
            
            Task:
            Let's think step by step.
            Step 1: Identify and explain any Environmental (E) aspects mentioned in the article.
            Environmental Aspects:

            Step 2: Based on Step 1, extract the original sentences from the article that relates to the Environmental Aspects. Return the sentences in an array.
            Environmental Array:

            Step 3: Identify and explain any Social (S) aspects mentioned in the article. 
            Social Aspects:

            Step 4: Based on Step 3, extract the original sentences from the article that relates to the Social Aspects. Return the sentences in an array.
            Social Array:
            
            Step 5: Identify and explain any Governance (G) aspects mentioned in the article.
            Governance Aspects:

            Step 6: Based on Step 5, extract the original sentences from the article that relates to the Governance Aspects. Return the sentences in an array.
            Governance Array:
    """

def generate_multiple_reasoning_paths(model, prompt, num_paths=3):
    paths = []
    for _ in range(num_paths):
        esg_sentence = model.extract_esg_sentence(prompt, temperature=0.7, verbose=False)
        paths.append(esg_sentence)
    return paths

def extract_and_intersect_combined(output_texts):
    combined_sets = set()
    categories = ["Environmental", "Social", "Governance"]
    for keyword in categories:
        extracted_sets = [set(utils.extract_json_array(output, keyword)) for output in output_texts]
        if extracted_sets:
            combined_sets.update(set.intersection(*extracted_sets))
    return combined_sets

def combine_and_deduplicate_combined(output_texts):
    combined_set = set()
    categories = ["Environmental", "Social", "Governance"]
    for keyword in categories:
        for output in output_texts:
            combined_set.update(utils.extract_json_array(output, keyword))
    return "\n".join(list(combined_set))

def main():
    model = GPT(system_context=system_context)
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = range(0, 21) 

    # Initialize a list to store the sentences and their corresponding ESG-related sentences
    data = []

    for index in rows_indices:
        row = df.iloc[index]
        prompt = create_prompt(row['title'], row['content'])
        results = {'Title': row['title'], 'URL': row['url']}
        print(results)
        
        # Generate multiple reasoning paths
        reasoning_paths = generate_multiple_reasoning_paths(model, prompt, num_paths=3)
        
        # Save each path as a separate column
        for j, path in enumerate(reasoning_paths):
            results[f'ESG Sentences Path {j+1}'] = path
        
        # Find the intersection of all paths for the combined ESG category
        ESG_intersection_combined = extract_and_intersect_combined(reasoning_paths)
        results['ESG Intersection Combined'] = "\n".join(list(ESG_intersection_combined))
        
        # Combine and deduplicate all sentences from all paths for the combined ESG category
        ESG_combined = combine_and_deduplicate_combined(reasoning_paths)
        results['ESG Combined'] = ESG_combined
        
        data.append(results)

    # Create a DataFrame from the data list
    result_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    result_df.to_csv("results/COT_SC_test.csv", index=False)

    # Example: Print the results for one row
    #output_texts = result_df.loc[0, 'ESG Sentences Path 1']
    #ESG_arr_combined = result_df.loc[0, 'ESG Combined']
    #ESG_arr_intersection_combined = result_df.loc[0, 'ESG Intersection Combined']

    #print("ESG Combined: ", ESG_arr_combined)
    #print("ESG Intersection Combined: ", ESG_arr_intersection_combined)

if __name__ == '__main__':
    main()