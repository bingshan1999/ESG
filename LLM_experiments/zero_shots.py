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

    rows_indices = range(0,21)
    # Split the first row's content into sentences
    #sentences = sent_tokenize(first_content)

    # Initialize a list to store the sentences and their corresponding ESG-related sentences
    data = []
    
    # Consider running average to reduce memory usage
    all_embeddings = {'Environmental': [], 'Social': [], 'Governance': []}
    
    for index in rows_indices:
        row = df.iloc[index]
        prompts = create_prompt(row['title'], row['content'])
        results = {'Title': row['title'], 'URL': row['url'], 'Environmental Cosine Similarity': 0, 'Social Cosine Similarity': 0, 'Governance Cosine Similarity': 0}
        print(results)

        for i, prompt in enumerate(prompts):
            esg_sentence = model.extract_esg_sentence(prompt, verbose=False)
            results[f'ESG Sentences Prompt {i+1}'] = esg_sentence
        
        esg_sentence = results['ESG Sentences Prompt 6']

        environmental_sentences, social_sentences, governance_sentences = utils.parse_esg_json(esg_sentence)

        if environmental_sentences:
            results['Environmental Cosine Similarity'] = utils.calculate_pairwise_cosine_similarity_str(environmental_sentences)
            all_embeddings['Environmental'].extend(utils.encode_arr(environmental_sentences))
        
        if social_sentences:
            results['Social Cosine Similarity'] = utils.calculate_pairwise_cosine_similarity_str(social_sentences)
            all_embeddings['Social'].extend(utils.encode_arr(social_sentences))

        if governance_sentences:
            results['Governance Cosine Similarity'] = utils.calculate_pairwise_cosine_similarity_str(governance_sentences)
            all_embeddings['Governance'].extend(utils.encode_arr(governance_sentences))

        data.append(results)

    #Create a DataFrame from the data list
    result_df = pd.DataFrame(data)

    #Save the DataFrame to a CSV file
    result_df.to_csv("results/zero_shots_test.csv", index=False)

    print(f"""
            Total embeddings:
            E: {len(all_embeddings['Environmental'])}
            S: {len(all_embeddings['Social'])}
            G: {len(all_embeddings['Governance'])}
            Overall Cosine similarity
            E: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['Environmental'])}
            S: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['Social'])}
            G: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['Governance'])}
        """)


if __name__ == '__main__':
    main()