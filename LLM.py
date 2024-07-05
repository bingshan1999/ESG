import os
import pandas as pd
from transformers import pipeline, GPT2Tokenizer
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from g4f.client import Client

# Set environment variable for debugging (only enable during debugging phase)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check if a GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load pre-trained GPT-Neo model and tokenizer using Hugging Face pipeline
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B', device=device)
# tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')

# generator = pipeline('text-generation', model='gpt2', device=device)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained GPT-Neo 1.3B model and tokenizer using Hugging Face pipeline
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=device)
# tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')

# Define the prompt template
prompt_template = (
    "Text: {sentence}\n"
    "Question: 1) Is the text provided above related to Environmental, Social, and Governance (ESG) topics? [YES/NO/UNSURE] "
    "2) If [YES], does it belongs to E,S, or G category? [E/S/G]. If [NO/UNSURE], put it as [UNKNOWN] " 
    "3) Justify your reasoning \n"
    "Answer the question in order and delimit by semicolon"
    
)

# Function to generate ESG-related sentences using GPT-Neo
# def extract_esg_sentence(sentence, num_sequences=1, max_new_tokens=250):
#     prompt = prompt_template.format(sentence=sentence)
#     print("prompt", prompt)
#     responses = generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=num_sequences, pad_token_id=generator.tokenizer.eos_token_id)
#     print("\n")
#     esg_sentence = responses[0]['generated_text'].replace(prompt, "").strip()  # Remove the prompt from the generated text
#     print("responses:", esg_sentence)
#     print("\n\n")
#     return esg_sentence

client = Client()
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": "Hello"}],
# )
#print(response.choices[0].message.content)

def extract_esg_sentence(sentence, num_sequences=1, max_retries=3, temperature=0.4, top_p=0.4):
    prompt = prompt_template.format(sentence=sentence)
    for _ in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=temperature,
            top_p=top_p,
            messages=[
                {"role": "system", "content": "You are an expert in Environmental, Social, and Governance (ESG) topics in cryptocurrency space. ESG might be related but not limited to carbon emission, waste produced, entry requirements, users, volatility, liquidity, grants, voting protocols, DAO, security risks and so on."},
                {"role": "user", "content": prompt}
            ],
        )
        esg_sentence = response.choices[0].message.content  # Extract the content from the response
        
        if not esg_sentence.lower().startswith("sorry,"):
            print(f"====================\nprompt: {prompt}\n\nResponse: {esg_sentence}")
            return esg_sentence
    
    return "Failed to get a response after multiple attempts"

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
sentences = sent_tokenize(first_content)

# Initialize a list to store the sentences and their corresponding ESG-related sentences
data = []

# Process each sentence in the first row's content
for sentence in sentences[:10]:
    esg_sentence = extract_esg_sentence(sentence)
    data.append({'sentence': sentence, 'esg_sentence': esg_sentence})

# Create a new DataFrame from the data list
df_first_row_sentences = pd.DataFrame(data)

# Save the new DataFrame to a CSV file
# output_file_path = 'data/first_row_sentences_with_esg.csv'
# df_first_row_sentences.to_csv(output_file_path, index=False)

# Set pandas option to display full text
#pd.set_option('display.max_colwidth', None)

# Print the new DataFrame
#print(df_first_row_sentences)