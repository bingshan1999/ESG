import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT

#nltk.download('punkt')
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

def create_prompt(title, content):
    return [
        #1 no guidelines
        f"""
            Article Title: {title}
            Article Context: {content}
            Task: {task}
        """,

        #2 normal with guidelines
        f"""
            Article Title: {title}
            Article Context: {content}
            Task: {task}
            Guidelines: {guidelines}
        """,

        #3 put tasks last
        f"""
            Article Title: {title}
            Article Context: {content}
            Guidelines: {guidelines}
            Task: {task}
        """,
        
        #4 No title
        f"""
            Article: {content}
            Task: {task}
            Guidelines: {guidelines}
        """,

        #5 Task at beginning
        f"""
            Task: {task}
            Article Title: {title}
            Article Context: {content}
            Guidelines: {guidelines}
        """,

        #6 Task and guidelines at the beginning
        f"""
            Task: {task}
            Guidelines: {guidelines}
            Article Title: {title}
            Article Context: {content}
        """
    ]

def main():
    model = GPT()
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = [0,20]
    # Split the first row's content into sentences
    #sentences = sent_tokenize(first_content)

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