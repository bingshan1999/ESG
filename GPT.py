import os
import pandas as pd
from transformers import pipeline, GPT2Tokenizer
from datasets import load_dataset, Dataset
import torch
from g4f.client import Client

# Set environment variable for debugging (only enable during debugging phase)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# # Check if a GPU is available
# device = 0 if torch.cuda.is_available() else -1

class GPT:
    def __init__(self, model_name='gpt-3.5-turbo', system_context=None):
        self.client = Client()
        self.model_name = model_name
        self.system_context = system_context 
    
    def extract_esg_sentence(self, prompt, max_retries=3, temperature=0.4, top_p=0.4, verbose=False):
        messages = []
        if self.system_context:
            messages.append({"role": "system", "content": self.system_context})
    
        messages.append({"role": "user", "content": prompt})
    
        for _ in range(max_retries):
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=temperature,
                top_p=top_p,
                messages=messages
            )
            esg_sentence = response.choices[0].message.content  # Extract the content from the response
            
            if not esg_sentence.lower().startswith("sorry,"):
                if verbose:
                    print(f"====================\nprompt: {prompt}\n\nResponse: {esg_sentence}")
                return esg_sentence
        
        return "Failed to get a response after multiple attempts"