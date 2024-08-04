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
import utils

#nltk.download('punkt')
guidelines = """
ESG topics in crypto can be related but not limit to:
- Environmental (E): aspects Energy consumption, resource management, renewable energy usage.
- Social (S) aspects: Labor practices, community engagement and inclusion, diversity and inclusion, security and user protection.
- Governance (G) aspects: Decentralized governance models, business ethics and transparency, regulatory compliance, executive compensation and incentives.
"""

task = """
1) Is the above sentence related to ESG issues? [YES/NO]
2) Justify your answer.
"""

def create_prompt(title, content):
    return [
        f"""
            Article Title: {title}
            Article Context: {content}
            Task: {task}
        """,
    ]

def main():
    model = GPT()
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = [0, 1]

    # Initialize a list to store the sentences and their corresponding ESG-related sentences
    data = []

    for index in rows_indices:
        row = df.iloc[index]
        sentences = sent_tokenize(row['content'])

        for sentence in sentences:
            prompt = create_prompt(row['title'], sentence)
            esg_sentence = model.extract_esg_sentence(prompt, verbose=False)
            data.append({'Title': row['title'], 'URL': row['url'], 'Sentence': sentence, 'Response': esg_sentence})

    # Create a DataFrame from the data list
    result_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    result_df.to_csv("results/sentences_test.csv", index=False)

if __name__ == '__main__':
    main()