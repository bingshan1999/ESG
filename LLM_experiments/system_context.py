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
    return [
        f"""
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
        """,

        #2 normal with guidelines
        f"""
            Article Title: {title}
            Article Context: {content}
            Task: {task}
            Guidelines: {guidelines}
        """]


system_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. Here are some examples and reasons for each ESG aspect in this context:

- Environmental (E):
  - Energy Consumption and Carbon Emissions: Bitcoin mining and other proof-of-work cryptocurrencies consume significant energy, raising concerns about sustainability and environmental impact.
  - Resource Management: The production and disposal of mining hardware can lead to electronic waste, necessitating proper resource management.
  - Renewable Energy Usage: Some projects are adopting renewable energy for mining operations to reduce their environmental footprint.

- Social (S):
  - Labor Practices: Fair labor practices and working conditions for those employed in the crypto industry are essential for social sustainability.
  - Community Engagement and Inclusion: Active and inclusive community engagement supports the development and adoption of crypto projects.
  - Diversity and Inclusion: Promoting diversity within teams and communities leads to more innovative and equitable solutions.
  - Security and User Protection: Prioritizing the security of users and their assets is a significant social responsibility in the crypto space.

- Governance (G):
  - Decentralized Governance Models: DAOs and other decentralized governance structures must be transparent, fair, and effective.
  - Business Ethics and Transparency: Ethical practices and clear communication build trust with stakeholders.
  - Regulatory Compliance: Adhering to AML, KYC, and other regulations is crucial for the legitimacy and sustainability of crypto projects.
  - Executive Compensation and Incentives: Fair and transparent compensation practices impact decision-making and project priorities.

Using these examples, you will evaluate the text provided for its relevance to each ESG aspect. Justify your reasoning for each evaluation.
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
  result_df.to_csv("results/system_context_test2.csv", index=False)

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