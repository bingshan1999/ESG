import pandas as pd
from transformers import pipeline
import torch

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT

import utils

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

ground_truth_path = '../data/ground_truth.csv'
ground_truth_df = pd.read_csv(ground_truth_path)

model = GPT(system_context=utils.system_context)
esg_bert_classifier = pipeline("text-classification", model="nbroad/ESG-BERT", device=device)
fin_bert_classifier = pipeline("text-classification", model="yiyanghkust/finbert-esg", device=device)

detailed_data = []
def create_prompt(sentence):
    return f"""
            The following sentence may or may not be related to Environmental, Social, or Governance (ESG) issues.

            Sentence: "{sentence}"
            
            On a scale of 0 to 1, where 0 means "not related at all" and 1 means "definitely related," how confident are you that this sentence belongs to an ESG issue? 
            Provide the confidence score up to 4 decimal places and nothing else.
            """

for index, row in ground_truth_df.iterrows():
    sentence = row['sentence']
    
    # Classify the sentence with each model
    gpt_response = model.extract_esg_sentence(create_prompt(sentence), verbose=False)
    esg_bert_result = esg_bert_classifier(sentence)[0]
    fin_bert_result = fin_bert_classifier(sentence)[0]

    detailed_data.append({
        'Sentence': sentence,
        'ESG-BERT Label': esg_bert_result['label'],
        'ESG-BERT Score': 1 - esg_bert_result['score'] if esg_bert_result['label'] == 'None' else esg_bert_result['score'],
        'FinBERT Label': fin_bert_result['label'],
        'FinBERT Score': 1 - fin_bert_result['score'] if fin_bert_result['label'] == 'None' else fin_bert_result['score'],
        'GPT Score': gpt_response
    })
            
# Convert the detailed data to a DataFrame
detailed_df = pd.DataFrame(detailed_data)

# Save the detailed DataFrame to a CSV file
detailed_df.to_csv('results/baseline.csv', index=False)

