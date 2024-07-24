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

def create_prompt(title, content):
    return f"""
        Article Title: {title}
        Article Context: {content}

        Step 1: Identify and explain any Environmental (E) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array..
        Environmental Aspects:
        Environmental Array:

        Step 2: Identify and explain any Social (S) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
        Social Aspects:
        Social Array:

        Step 3: Identify and explain any Governance (G) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
        Governance Aspects:
        Governance Array:
    """

def generate_multiple_reasoning_paths(model, prompt, num_paths=3):
    paths = []
    for _ in range(num_paths):
        esg_sentence = model.extract_esg_sentence(prompt, temperature=0.7, verbose=False)
        paths.append(esg_sentence)
    return paths

def extract_and_intersect(output_texts, keyword):
    extracted_sets = [set(utils.extract_json_array(output, keyword)) for output in output_texts]
    if extracted_sets:
        return set.intersection(*extracted_sets)
    return set()

def main():
    model = GPT()
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = [0, 20]

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
        
        # Find the intersection of all paths and save it as a separate column
        E_intersection = extract_and_intersect(reasoning_paths, "Environmental")
        S_intersection = extract_and_intersect(reasoning_paths, "Social")
        G_intersection = extract_and_intersect(reasoning_paths, "Governance")
        
        results['Environmental Intersection'] = list(E_intersection)
        results['Social Intersection'] = list(S_intersection)
        results['Governance Intersection'] = list(G_intersection)
        
        data.append(results)

    # Create a DataFrame from the data list
    result_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    result_df.to_csv("results/COT_SC_test.csv", index=False)

    # Example: Print the results for one row
    output_texts = result_df.loc[0, 'ESG Sentences Path 1']
    E_arr = utils.extract_json_array(output_texts, "Environmental")
    S_arr = utils.extract_json_array(output_texts, "Social")
    G_arr = utils.extract_json_array(output_texts, "Governance")

    print(f'Env arr: {E_arr} \nSocial arr: {S_arr}\nGov arr: {G_arr}')

    E_arr_intersection = result_df.loc[0, 'Environmental Intersection']
    S_arr_intersection = result_df.loc[0, 'Social Intersection']
    G_arr_intersection = result_df.loc[0, 'Governance Intersection']

    print("Environmental Intersection: ", E_arr_intersection)
    print("Social Intersection: ", S_arr_intersection)
    print("Governance Intersection: ", G_arr_intersection)

if __name__ == '__main__':
    main()