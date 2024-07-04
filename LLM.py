import os
import pandas as pd
from transformers import pipeline, GPT2Tokenizer
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm

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
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=device)
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')

# Define the prompt template
prompt_template = (
    "Identify the first sentence that is related to Environmental, Social, and Governance (ESG) topics in the following text:\n\n"
    "Text: {chunk}\n\n"
    "ESG-related sentence:"
)

# Function to split text into non-overlapping chunks considering the prompt length and max new tokens
def split_text(text, max_length=2048, max_new_tokens=150):
    prompt_length = len(tokenizer.encode(prompt_template.format(chunk="")))
    max_chunk_length = max_length - prompt_length - max_new_tokens
    tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
    chunks = [tokens[i:i + max_chunk_length] for i in range(0, len(tokens), max_chunk_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Function to generate ESG-related sentences using GPT-Neo
def extract_esg_sentences(text, num_sequences=3, max_new_tokens=150):
    chunks = split_text(text, max_new_tokens=max_new_tokens)
    esg_sentences = []
    for chunk in chunks:
        prompt = prompt_template.format(chunk=chunk)
        responses = generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=num_sequences, pad_token_id=generator.tokenizer.eos_token_id)
        esg_sentences.extend([response['generated_text'] for response in responses])
    return esg_sentences

# Load your data using datasets library
dataset = load_dataset('csv', data_files={'data': 'data/coindesk_btc.csv'})['data']

# Exclude rows with missing values in the 'content' column
dataset = dataset.filter(lambda x: x['content'] is not None)

# Limit dataset to the first 10 rows
dataset = dataset.select(range(10))

# Reduce batch size to manage memory usage
batch_size = 1

# Map the extract_esg_sentences function to the dataset with a progress bar
def map_function(batch):
    esg_sentences = []
    for text in tqdm(batch['content'], desc="Processing"):
        esg_sentences.append(extract_esg_sentences(text))
    batch['esg_sentences'] = esg_sentences
    return batch

# Apply the map function to the dataset in smaller batches and clear GPU cache
for i in range(0, len(dataset), batch_size):
    subset = dataset.select(range(i, min(i + batch_size, len(dataset))))
    subset = subset.map(map_function, batched=True, batch_size=batch_size)
    
    # Convert subset back to pandas DataFrame and append to df_cleaned
    if i == 0:
        df_cleaned = subset.to_pandas()
    else:
        df_cleaned = pd.concat([df_cleaned, subset.to_pandas()], ignore_index=True)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Display the first few rows with ESG-related sentences
print(df_cleaned[['content', 'esg_sentences']].head())