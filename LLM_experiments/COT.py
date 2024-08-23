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

guidelines = """
ESG topics in crypto can be related but not limit to:
- Environmental (E): aspects Energy consumption, resource management, renewable energy usage.
- Social (S) aspects: Labor practices, community engagement and inclusion, diversity and inclusion, security and user protection.
- Governance (G) aspects: Decentralized governance models, business ethics and transparency, regulatory compliance, executive compensation and incentives.
"""

def create_prompt(title, content):
    return [
        # 1. Summary
        f"""
            Article Title: {title}
            Article Context: {content}
            
            Step 1: Summarize the article.

            Step 2: Identify and explain any Environmental (E) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array..
            Environmental Aspects:
            Environmental Array:

            Step 3: Identify and explain any Social (S) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
            Social Aspects:
            Social Array:

            Step 4: Identify and explain any Governance (G) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
            Governance Aspects:
            Governance Array:
        """,       
        f"""
            Article Title: {title}
            Article Context: {content}
            
            Step 2: Identify and explain any Environmental (E) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array..
            Environmental Aspects:
            Environmental Array:

            Step 3: Identify and explain any Social (S) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
            Social Aspects:
            Social Array:

            Step 4: Identify and explain any Governance (G) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
            Governance Aspects:
            Governance Array:
        """,
        
        # 2. No title 
        f"""
            Article Context: {content}
            
            Step 1: Identify and explain any Environmental (E) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array..
            Environmental Aspects:
            Environmental Array:

            Step 2: Identify and explain any Social (S) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
            Social Aspects:
            Social Array:

            Step 3: Identify and explain any Governance (G) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
            Governance Aspects:
            Governance Array:
            """,

        # 3. More steps (separating aspect and array)
        f"""
            Article Title: {title}
            Article Context: {content}

            Step 1: Identify and explain any Environmental (E) aspects mentioned in the article.
            Environmental Aspects:

            Step 2: Based on Step 1, extract the original sentences from the article that relates to the Environmental Aspects. Return the sentences in an array.
            Environmental Array:

            Step 3: Identify and explain any Social (S) aspects mentioned in the article. 
            Social Aspects:

            Step 4: Based on Step 3, extract the original sentences from the article that relates to the Social Aspects. Return the sentences in an array.
            Social Array:
            
            Step 5: Identify and explain any Governance (G) aspects mentioned in the article.
            Governance Aspects:

            Step 6: Based on Step 5, extract the original sentences from the article that relates to the Governance Aspects. Return the sentences in an array.
            Governance Array:
        """,
        
        # 4. Lesser steps (combining aspect and array)
        f"""
            Article Title: {title}
            Article Context: {content}

            Step 1: Identify and explain any Environmental (E) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
            Environmental Aspects:
            Environmental Array:

            Step 2: Identify and explain any Social (S) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
            Social Aspects:
            Social Array:

            Step 3: Identify and explain any Governance (G) aspects mentioned in the context. Additionally, extract the relevant sentences from the article and return it as an array.
            Governance Aspects:
            Governance Array:
        """
    ]

def main():
    model = GPT()
    # Load your data using pandas
    file_path = '../data/coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = range(0, 5) 

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
        
        esg_sentence = results['ESG Sentences Prompt 5']

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
    result_df.to_csv("results/COT_test_uncleaned.csv", index=False)
    print(all_embeddings)
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