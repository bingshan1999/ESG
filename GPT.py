import os
from transformers import pipeline, GPT2Tokenizer
from datasets import load_dataset, Dataset
import torch
from g4f.client import Client
from g4f.Provider.Bing import Bing
from g4f.Provider.Chatgpt4o import Chatgpt4o
from g4f.Provider.FreeChatgpt import FreeChatgpt
from g4f.Provider.Chatgpt4Online import Chatgpt4Online
from g4f.Provider.ChatgptFree import ChatgptFree
import time
# Set environment variable for debugging (only enable during debugging phase)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# # Check if a GPU is available
# device = 0 if torch.cuda.is_available() else -1

import os.path
from g4f.cookies import set_cookies_dir, read_cookie_files

import g4f.debug
g4f.debug.logging = True

# cookies_dir = os.path.join(os.path.dirname(__file__), "har_and_cookies")
# set_cookies_dir(cookies_dir)
# read_cookie_files(cookies_dir)

model = Chatgpt4o()
model.__name__ = "Chatgpt4Online"

#print(dir(model))


class GPT:
    def __init__(self, model_name='gpt-4', system_context=None, provider=model):
        self.client = Client()
        self.model_name = model_name
        self.system_context = system_context 
        self.provider= provider
    
    '''
    temperature: ranges from 0 to 1 where 0=deterministic (no randomness) and 1=highest randomness in the output
    top_p: nucleus sampling. Work with temperature to randomly choose cumulative probability of p% of words.
    '''
    def extract_esg_sentence(self, prompt, max_retries=3, temperature=0, top_p=0.4, verbose=False):
        #print(f'temperature: {temperature}')
        messages = []
        if self.system_context:
            messages.append({"role": "system", "content": self.system_context})
    
        messages.append({"role": "user", "content": prompt})
        
        for retry in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    top_p=top_p,
                    provider=self.provider,
                    messages=messages
                )
                esg_sentence = response.choices[0].message.content  # Extract the content from the response
                
                if not esg_sentence.lower().startswith("sorry,"):
                    if verbose:
                        print(f"====================\nprompt: {prompt}\n\nResponse: {esg_sentence}")                                    
                        # with open('output.txt', 'w') as file:
                        #     print("writing to file")
                        #     file.write(esg_sentence)

                    return esg_sentence
            
            except Exception as e:
                if "429" in str(e):  # Check if the error message contains "429"
                    print(f"Rate limit reached. Retrying in 30 seconds... (Attempt {retry + 1} of {max_retries})")
                    time.sleep(30)  # Wait for 60 seconds before retrying
                else:
                    print(f"GPT failed with error: {e}. (Attempt {retry + 1} of {max_retries})")
     
        
        return "Failed to get a response after multiple attempts"