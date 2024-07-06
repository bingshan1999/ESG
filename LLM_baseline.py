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

model = GPT('gpt-3.5-turbo')
# Load your data using pandas
file_path = 'data/coindesk_btc.csv'
df = pd.read_csv(file_path)

# Exclude rows with missing values in the 'content' column
df_cleaned = df.dropna(subset=['content'])

# Limit the dataset to the first 10 rows for testing
df_cleaned = df_cleaned.head(10)

# Get the first row's content
first_content = df_cleaned.iloc[0]['content']

# Split the first row's content into sentences
#sentences = sent_tokenize(first_content)

# Initialize a list to store the sentences and their corresponding ESG-related sentences
data = []

# Process each sentence in the first row's content
# for sentence in sentences[:10]:
#     esg_sentence = extract_esg_sentence(sentence)
#     data.append({'sentence': sentence, 'esg_sentence': esg_sentence})

esg_sentence = model.extract_esg_sentence(first_content, verbose=True)

#data.append({'sentence': sentence, 'esg_sentence': esg_sentence})
# Create a new DataFrame from the data list
#df_first_row_sentences = pd.DataFrame(data)

# Save the new DataFrame to a CSV file
# output_file_path = 'data/first_row_sentences_with_esg.csv'
# df_first_row_sentences.to_csv(output_file_path, index=False)

# Set pandas option to display full text
#pd.set_option('display.max_colwidth', None)

# Print the new DataFrame
#print(df_first_row_sentences)



# Select the most coherent response
# selected_aspect, best_response = select_most_coherent_response(responses)
# print(f"Selected Aspect: {selected_aspect}\nBest Response: {best_response}")