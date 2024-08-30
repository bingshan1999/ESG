import pandas as pd
from transformers import pipeline
import torch
import nltk
nltk.download('punkt')
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

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production, HPC.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection (Hacks), Entry Barrier and Accessibility (Global Reach, User Adoptions, Investment), Market Instability (Price Drops and Increases), Illicit Activities, Large Financial Institutions and Crypto Institution
- Governance (G): Decentralized Governance Models (Off-chain and On-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
"""

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load your data using pandas
file_path = '../data/cleaned_coindesk_btc.csv'
df = pd.read_csv(file_path)

ground_truth_path = '../data/ground_truth.csv'
ground_truth = utils.read_ground_truth_from_csv(ground_truth_path)

rows_indices = [i for i in range(0, 22) if i not in [6, 14]]  # Exclude specific articles
#rows_indices = [0,1]
# Create a subset of the DataFrame using the specified row indices
subset_df = df.iloc[rows_indices].copy()

model = GPT(system_context=system_context)
esg_classifier = pipeline("text-classification", model="nbroad/ESG-BERT", device=device)
#esg_classifier = pipeline("text-classification", model="yiyanghkust/finbert-esg", device=device)

# Function to classify each sentence in an article
def classify_sentences(article_text):
    sentences = sent_tokenize(article_text)
    classifications = []
    for sentence in sentences:
        result = esg_classifier(sentence)
        classifications.append(result)
    return sentences, classifications

def create_prompt(sentence):
    return f"""
            The following sentence may or may not be related to Environmental, Social, or Governance (ESG) issues.

            Sentence: "{sentence}"
            
            On a scale of 0 to 1, where 0 means "not related at all" and 1 means "definitely related," how confident are you that this sentence belongs to an ESG issue? 
            Provide the confidence score up to 4 decimal places and nothing else.
            """

# Create a detailed DataFrame for sentence classifications
detailed_data = []

#esg_sentence = model.extract_esg_sentence(prompt, verbose=False)
# Apply the classification to each sentence in the subset DataFrame
for index, row in subset_df.iterrows():
    article_ground_truth = ground_truth[index + 1]
    sentences, classifications = classify_sentences(row['content'])
    
    # Preprocess ground truth sentences for IoU comparison
    ground_truth_sets = [utils.preprocess_sentence(gt) for gt in article_ground_truth]

    for sentence, classification in zip(sentences, classifications):
        sentence_set = utils.preprocess_sentence(sentence)
        is_match = False 
        
        # Check IoU against all ground truth sentences
        for gt_set in ground_truth_sets:
            iou = utils.calculate_iou(sentence_set, gt_set)
            if iou >= 0.5:
                is_match = True
                break

        if is_match:
            GPT_ans = model.extract_esg_sentence(create_prompt(sentence), verbose=False)
            detailed_data.append({
                'Title': row['title'],
                #'URL': row['url'],
                'Sentence': sentence,
                'Label': classification[0]['label'],
                'Score': classification[0]['score'],
                'GPT Score': GPT_ans
            })
            
# Convert the detailed data to a DataFrame
detailed_df = pd.DataFrame(detailed_data)

# Save the detailed DataFrame to a CSV file
detailed_df.to_csv('results/baseline.csv', index=False)

# Print the detailed DataFrame
#print(detailed_df)
