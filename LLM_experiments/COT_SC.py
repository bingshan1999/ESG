import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

import random
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT
import utils

random.seed(42)

system_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production, HPC.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection (Hacks), Entry Barrier and Accessibility (Global Reach, User Adoptions, Investment), Market Instability (Price Drops and Increases), Illicit Activities, Large Financial Institutions and Crypto Institution
- Governance (G): Decentralized Governance Models (Off-chain and On-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
"""

def create_prompt(title, content):
    return f"""
                Article Title: {title}
                Article Context: {content}

                Task:
                Let's think step by step.
                Step 1: Identify and explain any Environmental (E) aspects mentioned in the article.
                Environmental Aspects:

                Step 2: Based on Step 1, extract the original sentences from the article that relates to the Environmental Aspects. Return the sentences in a JSON array.
                Environmental Array:

                Step 3: Identify and explain any Social (S) aspects mentioned in the article. 
                Social Aspects:

                Step 4: Based on Step 3, extract the original sentences from the article that relates to the Social Aspects. Return the sentences in a JSON array.
                Social Array:
                
                Step 5: Identify and explain any Governance (G) aspects mentioned in the article.
                Governance Aspects:

                Step 6: Based on Step 5, extract the original sentences from the article that relates to the Governance Aspects. Return the sentences in a JSON array.
                Governance Array:
            """

def generate_multiple_reasoning_paths(model, prompt, num_paths=3):
    paths = []
    for _ in range(num_paths):
        esg_sentence = model.extract_esg_sentence(prompt, temperature=random.random(), verbose=False)
        paths.append(esg_sentence)
    return paths

def extract_and_intersect_combined(output_texts):
    combined_sets = set()
    for sent in output_texts:
        print(f'sent: {sent}')
        all_sentences = utils.parse_esg_json(sent)
        # Process sentences: Upper-case the first letter and add to the set
        processed_sentences = {sentence.capitalize() for sentence in all_sentences}
        combined_sets.update(processed_sentences)
    print(f'EXTRACTED: {combined_sets}')
    return list(combined_sets)

def main():
    model = GPT(system_context=system_context)
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    ground_truth_path = '../data/ground_truth.csv'
    ground_truth = utils.read_ground_truth_from_csv(ground_truth_path)

    rows_indices = [i for i in range(0, 22) if i not in [6, 14]]  # Exclude specific articles
    #rows_indices = [0,1]
    # Initialize a list to store the sentences and their corresponding ESG-related sentences
    data = []
    all_embeddings = []

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
        results['ESG Intersection Combined'] = "\n".join(ESG_intersection_combined)
        
        tp, fp, fn, all_iou, best_iou = utils.evaluate_extracted_sentences(ESG_intersection_combined, ground_truth[index + 1])
        print(f'tp: {tp}, fp: {fp}, fn: {fn}, all_iou: {all_iou:.4f}, best_iou: {best_iou:.4f}')
        #precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        #recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        all_embeddings.extend(utils.encode_arr(ESG_intersection_combined))
        # Store metrics for this agent count
        results['TP'] = tp
        results['FP'] = fp
        results['FN'] = fn
        results['All IOU'] = all_iou
        results['Best IOU'] = best_iou        
 
        data.append(results)

    # Create a DataFrame from the data list
    result_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    result_df.to_csv("results/COT_SC_test.csv", index=False)
    print(f'Overall Cosine Similarity: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings)}')

if __name__ == '__main__':
    main()