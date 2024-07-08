import pandas as pd
import torch
from tqdm import tqdm
from GPT import GPT
import nltk
from nltk.tokenize import sent_tokenize

#nltk.download('punkt')
# Define the prompt template
def create_prompt(text):
    return f"""
    Text: {text}
    
    Identify any sentences from the text that might involve ESG (Environmental, Social, Governance) topics related to Bitcoin. 

    ESG topics can be related but not limit to:
    Environmental (E): aspects Energy consumption, resource management, renewable energy usage.
    Social (S) aspects: Labor practices, community engagement and inclusion, diversity and inclusion, security and user protection.
    Governance (G) aspects: Decentralized governance models, business ethics and transparency, regulatory compliance, executive compensation and incentives.

    
    Based on the above evaluations, return a JSON object Following these key-value pairs 
    1) 'Environmental': An array containing all sentences related to Environmental aspect 
    2) 'Social': An array containing all sentences related to Social aspect 
    3) 'Governance':  An array containing all sentences related to Governance aspect 
    4) 'Reason': A summarized justification of your evaluation.
    """

model = GPT('gpt-3.5-turbo')
# Load your data using pandas
file_path = '../data/coindesk_btc.csv'
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

esg_sentence = model.extract_esg_sentence(create_prompt(first_content), verbose=True)

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