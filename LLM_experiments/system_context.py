import pandas as pd
import torch
from tqdm import tqdm
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT

guidelines = """
ESG topics in crypto can be related but not limit to:
- Environmental (E): aspects Energy consumption, resource management, renewable energy usage.
- Social (S) aspects: Labor practices, community engagement and inclusion, diversity and inclusion, security and user protection.
- Governance (G) aspects: Decentralized governance models, business ethics and transparency, regulatory compliance, executive compensation and incentives.
"""

task = """
Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics related to Bitcoin. 
Return a JSON object Following these key-value pairs and nothing else
1) 'Environmental': An array containing all sentences related to Environmental aspect 
2) 'Social': An array containing all sentences related to Social aspect 
3) 'Governance':  An array containing all sentences related to Governance aspect 
"""

# Define the prompt template
def create_prompt(title, content):
    return [f"""
          Article Title: {title}
          Article Context: {content}
          
          Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics related to Bitcoin. 
          Return a JSON object Following these key-value pairs and nothing else
          1) 'Environmental': An array containing all sentences related to Environmental aspect 
          2) 'Social': An array containing all sentences related to Social aspect 
          3) 'Governance':  An array containing all sentences related to Governance aspect 
        """,
        f"""
            Article Title: {title}
            Article Context: {content}

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
        """]


system_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection, Entry Barrier and Accessibility, Market Instability, Illicit Activities, Influence of major financial institutions
- Governance (G): Decentralized Governance Models (off-chain and on-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges

"""

def main():
  model = GPT(system_context=system_context)

  # Load your data using pandas
  file_path = '../data/cleaned_coindesk_btc.csv'
  df = pd.read_csv(file_path)

  rows_indices = [0,20]

  # Initialize a list to store the sentences and their corresponding ESG-related sentences
  data = []

  for index in rows_indices:
      row = df.iloc[index]
      prompts = create_prompt(row['title'], row['content'])
      results = {'Title': row['title'], 'URL': row['url']}
      print(results)

      for i, prompt in enumerate(prompts):
          esg_sentence = model.extract_esg_sentence(prompt, verbose=False)
          results[f'ESG Sentences Prompt {i+1}'] = esg_sentence
      
      data.append(results)


  #Create a DataFrame from the data list
  result_df = pd.DataFrame(data)

  #Save the DataFrame to a CSV file
  result_df.to_csv("results/system_context_test.csv", index=False)

if __name__ == '__main__':
    main()
#######################################
# NER WITH SPACY
#import spacy

## Load spaCy's pre-trained NER model
#nlp = spacy.load("en_core_web_sm")
# # def extract_entities(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# def highlight_entities(text, entities):
#     highlighted_text = text
#     for entity, label in entities:
#         highlighted_text = highlighted_text.replace(entity, f"[{label}: {entity}]")
#     return highlighted_text

# row = df.iloc[0]
# print("Extracted entities", extract_entities(row['content']))

# print("Highlighted Text:", highlight_entities(row['content'], extract_entities(row['content'])))
#######################################