import pandas as pd
import torch
from tqdm import tqdm
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT
import utils

# system_context = """
# You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
# Given an article, you will be asked to extract ESG issues from it. 
# Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

# - Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production, HPC.
# - Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection (Hacks), Entry Barrier and Accessibility (Global Reach, User Adoptions, Investment), Market Instability (Price Drops and Increases), Illicit Activities, Large Financial Institutions and Crypto Institution
# - Governance (G): Decentralized Governance Models (Off-chain and On-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
# """

system_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection, Entry Barrier and Accessibility, Market Instability, Illicit Activities, Influence of Large Financial Institutions and Crypto Institution
- Governance (G): Decentralized Governance Models (off-chain and on-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
"""

system_context_2 = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Consensus Algorithm, particularly Proof of Work that are energy-intensive
- E-waste caused by hardware updates, leading to short tech lifespans
- Carbon footprint of energy sources used
- On-chain governance and Off-chain governance structure
- Regulation including taxation, legal authority
- KYC/AML
- Geographical differences that leads to financial inclusivity and regulations
- Market Instability including huge drop in price, constituting risks to investor
- Illicit activities for example fraud, corruption, financial crimes
"""
# Define the prompt template
def non_system_context_prompt(title, content):
    return [f"""
          {system_context}
          Article Title: {title}
          Article Context: {content}
          
          Task: Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics. 
          Return your answer in a JSON array format with each identified sentence as a string.
        """,
        f"""
            {system_context}
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
        """]

def system_context_prompt(title, content):
    return [f"""
          Article Title: {title}
          Article Context: {content}
          
          Task: 
          Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics related to Bitcoin. 
          Return your answer in a JSON array format with each identified sentence as a string.
        """,
        f"""
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
        """]

def read_ground_truth_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    ground_truth_by_article = df.groupby('id')['sentence'].apply(list).to_dict()
    return ground_truth_by_article

def main():
    model = GPT(system_context=system_context)
    model_2 = GPT()

    # Load your data using pandas
    data_file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(data_file_path)

    ground_truth_path = '../data/ground_truth.csv'
    ground_truth = read_ground_truth_from_csv(ground_truth_path)

    rows_indices = [i for i in range(0, 22) if i not in [6, 14]]  # Exclude specific articles

    all_sentence_embeddings = []
    data = []

    for index in rows_indices:
        row = df.iloc[index]
        prompts_1 = system_context_prompt(row['title'], row['content'])
        prompts_2 = non_system_context_prompt(row['title'], row['content'])
        results = {'Title': row['title'], 'URL': row['url']}
        print(f'{index}: {results}')

        # Loop over the system context prompts
        for i, prompt in enumerate(prompts_1):
            esg_sentences = model.extract_esg_sentence(prompt, verbose=False)
            results[f'ESG Sentences System Prompt {i+1}'] = esg_sentences

            # Get all sentences as a single list from parse_esg_json
            esg_sentence_list = utils.parse_esg_json(esg_sentences)

            # Encode sentences and add to global list
            embeddings = utils.encode_arr(esg_sentence_list)
            all_sentence_embeddings.extend(embeddings)

            # Evaluate sentences and get TP, FP, FN, IoU, Precision, Recall
            tp, fp, fn, overall_iou, token_coverage, precision, recall = utils.evaluate_extracted_sentences(esg_sentence_list, ground_truth[index+1])

            # Add these metrics to the results
            results[f'System Prompt {i+1} TP'] = tp
            results[f'System Prompt {i+1} FP'] = fp
            results[f'System Prompt {i+1} FN'] = fn
            results[f'System Prompt {i+1} IoU'] = overall_iou
            results[f'System Prompt {i+1} Coverage'] = token_coverage
            #results[f'System Prompt {i+1} Precision'] = precision
            #results[f'System Prompt {i+1} Recall'] = recall

        # Loop over the non-system context prompts
        for i, prompt in enumerate(prompts_2):
            esg_sentences = model_2.extract_esg_sentence(prompt, verbose=False)
            results[f'ESG Sentences Non-System Prompt {i+1}'] = esg_sentences

            # Get all sentences as a single list from parse_esg_json
            esg_sentence_list = utils.parse_esg_json(esg_sentences)

            # Encode sentences and add to global list
            embeddings = utils.encode_arr(esg_sentence_list)
            all_sentence_embeddings.extend(embeddings)

            # Evaluate sentences and get TP, FP, FN, IoU, Precision, Recall
            tp, fp, fn, overall_iou, token_coverage, precision, recall = utils.evaluate_extracted_sentences(esg_sentence_list, ground_truth[index+1])

            # Add these metrics to the results
            results[f'Non-System Prompt {i+1} TP'] = tp
            results[f'Non-System Prompt {i+1} FP'] = fp
            results[f'Non-System Prompt {i+1} FN'] = fn
            results[f'Non-System Prompt {i+1} IoU'] = overall_iou
            results[f'System Prompt {i+1} Coverage'] = token_coverage
            #results[f'Non-System Prompt {i+1} Precision'] = precision
            #results[f'Non-System Prompt {i+1} Recall'] = recall

        # Loop over the user context prompts
        for i, prompt in enumerate(prompts_1):
            esg_sentences = model_2.extract_esg_sentence(prompt, verbose=False)
            results[f'ESG Sentences User Prompt {i+1}'] = esg_sentences

            # Get all sentences as a single list from parse_esg_json
            esg_sentence_list = utils.parse_esg_json(esg_sentences)

            # Encode sentences and add to global list
            embeddings = utils.encode_arr(esg_sentence_list)
            all_sentence_embeddings.extend(embeddings)

            # Evaluate sentences and get TP, FP, FN, IoU, Precision, Recall
            tp, fp, fn, overall_iou, token_coverage, precision, recall = utils.evaluate_extracted_sentences(esg_sentence_list, ground_truth[index+1])

            # Add these metrics to the results
            results[f'User Prompt {i+1} TP'] = tp
            results[f'User Prompt {i+1} FP'] = fp
            results[f'User Prompt {i+1} FN'] = fn
            results[f'User Prompt {i+1} IoU'] = overall_iou
            results[f'System Prompt {i+1} Coverage'] = token_coverage
            #results[f'User Prompt {i+1} Precision'] = precision
            #results[f'User Prompt {i+1} Recall'] = recall

        data.append(results)

    # Create a DataFrame from the data list
    result_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    result_df.to_csv("results/updated_system_context_test_new.csv", index=False)

    print("Results saved to 'results/updated_system_context_test_new_2.csv'.")

    # Calculate cosine similarity of all extracted sentences
    overall_cosine_similarity = utils.calculate_pairwise_cosine_similarity_embedding(all_sentence_embeddings)
    print(f"Overall Cosine Similarity: {overall_cosine_similarity}")

if __name__ == "__main__":
    main()
