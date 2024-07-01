import pandas as pd
from transformers import pipeline
import torch

# Check if a GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load your data
file_path = 'data/coindesk_btc.csv'
df = pd.read_csv(file_path)

# Exclude rows with missing values in the 'content' column
df_cleaned = df.dropna(subset=['content'])

# Load pre-trained GPT-2 model and tokenizer using Hugging Face pipeline
generator = pipeline('text-generation', model='gpt2', device=device)
#generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
#generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')

# Function to generate ESG-related sentences using GPT-2
def extract_esg_sentences(text):
    # prompt = (
    #     "Identify sentences that are related to Environmental, Social, and Governance (ESG) topics in the following text:\n\n"
    #     f"Text: {text}\n\n"
    #     "ESG-related sentences:"
    # )

    prompt = (
        "Summarize this article: \n\n"
        f"Text: {text} \n\n"
        "Summary:"
    )
    
    response = generator(prompt, max_new_tokens=1024, num_return_sequences=1)
    esg_sentences = response[0]['generated_text']
    return esg_sentences

# Extract ESG-related sentences for each article
df_cleaned['esg_sentences'] = df_cleaned['content'].apply(extract_esg_sentences)

# Display the first few rows with ESG-related sentences
print(df_cleaned[['content', 'esg_sentences']].head())
