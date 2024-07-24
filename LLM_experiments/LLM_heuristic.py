import pandas as pd
import torch
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from GPT import GPT

# Define the prompt template
def create_prompt(text):
    return f"""
    Text: {text}
    
    First, consider the following Environmental (E) aspects: Energy consumption, resource management, renewable energy usage.
    Question: Is the text provided above related to Environmental (E) topics? Justify your reasoning:
    Answer the question in the following format: [YES/NO].[REASON]
    
    Next, consider the following Social (S) aspects: Labor practices, community engagement and inclusion, diversity and inclusion, security and user protection.
    Question: Is the text provided above related to Social (S) topics? Justify your reasoning:
    Answer the question in the following format: [YES/NO].[REASON]
    
    Finally, consider the following Governance (G) aspects: Decentralized governance models, business ethics and transparency, regulatory compliance, executive compensation and incentives.
    Question: Is the text provided above related to Governance (G) topics? Justify your reasoning:
    Answer the question in the following format: [YES/NO].[REASON]
    
    Based on the above evaluations, provide a final assessment of the text's relevance to ESG topics and justify your reasoning.
    """

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
  result_df.to_csv("results/zero_shots_test.csv", index=False)

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