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
        f"""
            Article Title: {title}
            Article Context: {content}
            
            Let's think step by step.

            Step 1: Summarize the context based on the title.
            Summary: 

            Step 2: Identify and explain any Environmental (E) aspects mentioned in the context.
            Environmental Aspects:

            Step 3: Identify and explain any Social (S) aspects mentioned in the context.
            Social Aspects:

            Step 4: Identify and explain any Governance (G) aspects mentioned in the context.
            Governance Aspects:

            Step 5: Justify the relevance of these aspects to ESG considerations.
            Justification:
            """,
        
        # 2. No Justification
        f"""
            Article Title: {title}
            Article Context: {content}
            
            Let's think step by step.

            Step 1: Summarize the context based on the title.
            Summary: 

            Step 2: Identify and explain any Environmental (E) aspects mentioned in the context.
            Environmental Aspects:

            Step 3: Identify and explain any Social (S) aspects mentioned in the context.
            Social Aspects:

            Step 4: Identify and explain any Governance (G) aspects mentioned in the context.
            Governance Aspects:
            """,

        # 3. No summary
        f"""
            Article Title: {title}
            Article Context: {content}
        
            Step 1: Identify and explain any Environmental (E) aspects mentioned in the context.
            Environmental Aspects:

            Step 2: Identify and explain any Social (S) aspects mentioned in the context.
            Social Aspects:

            Step 3: Identify and explain any Governance (G) aspects mentioned in the context.
            Governance Aspects:

            Step 4: Justify the relevance of these aspects to ESG considerations.
            Justification:
            """,
        
        # 4. No Summary and Justification
        f"""
            Article Title: {title}
            Article Context: {content}
            
            Step 1: Identify and explain any Environmental (E) aspects mentioned in the context. Try to retrain the original sentences from the article.
            Environmental Aspects:

            Step 2: Identify and explain any Social (S) aspects mentioned in the context. Try to retrain the original sentences from the article.
            Social Aspects:

            Step 3: Identify and explain any Governance (G) aspects mentioned in the context. Try to retrain the original sentences from the article.
            Governance Aspects:
            """,
        
        # 5. No title 
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

        # 6. More steps (separating aspect and array)
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
        
        # 7. Lesser steps (combining aspect and array)
        f"""
            Article Title: {title}
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
        """
    ]

def main():
    model = GPT()
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = range(0, 21) 

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

    # output_text = result_df.loc[0, 'ESG Sentences Prompt 5']
    # E_arr = utils.extract_json_array(output_text, "Environmental")
    # S_arr = utils.extract_json_array(output_text, "Social")
    # G_arr = utils.extract_json_array(output_text, "Governance")

    # output_text = result_df.loc[0, 'ESG Sentences Prompt 6']
    # E_arr_2 = utils.extract_json_array(output_text, "Environmental")
    # S_arr_2 = utils.extract_json_array(output_text, "Social")
    # G_arr_2 = utils.extract_json_array(output_text, "Governance")

    # print(f'Env arr: {E_arr} \nSocial arr: {S_arr}\nGov arr: {G_arr}')
    # print("\n\n")
    # print(f'Env arr2: {E_arr_2} \nSocial arr2: {S_arr_2}\nGov arr_2: {G_arr_2}')
    # print("\n\n")
    # print("intersecting: ", utils.lists_to_intersection(S_arr,S_arr_2))
    # Example: Calculate similarity between ESG Sentences from Prompt 1 and Prompt 2 in the same row
    # for index, row in result_df.iterrows():
    #     esg_sentence_1 = row.get('ESG Sentences Prompt 1', '')
    #     esg_sentence_2 = row.get('ESG Sentences Prompt 2', '')
        
    #     if esg_sentence_1 and esg_sentence_2:
    #         similarity = utils.calculate_similarity(esg_sentence_1, esg_sentence_2)
    #         result_df.at[index, 'Similarity Prompt 1 and 2'] = similarity


    #Save the DataFrame to a CSV file
    result_df.to_csv("results/COT_test.csv", index=False)

if __name__ == '__main__':
    main()