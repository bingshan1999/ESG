import pandas as pd
import torch
from tqdm import tqdm
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT
import utils

system_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection, Entry Barrier and Accessibility, Market Instability, Illicit Activities, Influence of major financial institutions
- Governance (G): Decentralized Governance Models (off-chain and on-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
"""

system_context_2 = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Consensus Algorithm, particularly Proof of Work that are energy-intensive
- E-waste caused by hardware updates, leading to short tech lifespans
- Carbon footprint of energy sources used
- On-chain governance and Off-chain governance structure
- Regulation including taxation, legal authority
- KYC/AML
- Geographical differences that leads to financial inclusivity and regulations
- Market Instability including huge drop in price, constituting risks to investor
- Illicit activities for example fraud, corruption, financial crimes
"""
# Define the prompt template
def non_system_context_prompt(title, content):
    return [f"""
          {system_context_2}
          Article Title: {title}
          Article Context: {content}
          
          Task: Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics. 
          Return a JSON object Following these key-value pairs and nothing else
          1) 'Environmental': An array containing all sentences related to Environmental aspect 
          2) 'Social': An array containing all sentences related to Social aspect 
          3) 'Governance':  An array containing all sentences related to Governance aspect 
        """,
        f"""
            {system_context_2}
            Article Title: {title}
            Article Context: {content}

            Task:
            Let's think step by step.
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
        """]

def system_context_prompt(title, content):
    return [f"""
          Article Title: {title}
          Article Context: {content}
          
          Task: 
          Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics related to Bitcoin. 
          Return a JSON object Following these key-value pairs and nothing else
          1) 'Environmental': An array containing all sentences related to Environmental aspect 
          2) 'Social': An array containing all sentences related to Social aspect 
          3) 'Governance':  An array containing all sentences related to Governance aspect 
        """,
        f"""
            Article Title: {title}
            Article Context: {content}

            Task:
            Let's think step by step.
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
        """]

def main():
    model = GPT(system_context=system_context_2)
    model_2 = GPT()

    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = range(0, 21)

    # Initialize a list to store the sentences and their corresponding ESG-related sentences
    data = []
    # Consider running average to reduce memory usage
    all_embeddings = {
        'System Context Prompt 1': {'Environmental': [], 'Social': [], 'Governance': []},
        'System Context Prompt 2': {'Environmental': [], 'Social': [], 'Governance': []},
        'User Context Prompt 1': {'Environmental': [], 'Social': [], 'Governance': []},
        'User Context Prompt 2': {'Environmental': [], 'Social': [], 'Governance': []},
        'Non-System Context Prompt 1': {'Environmental': [], 'Social': [], 'Governance': []},
        'Non-System Context Prompt 2': {'Environmental': [], 'Social': [], 'Governance': []}
    }

    for index in rows_indices:
        row = df.iloc[index]
        prompts_1 = system_context_prompt(row['title'], row['content'])
        prompts_2 = non_system_context_prompt(row['title'], row['content'])
        results = {'Title': row['title'], 'URL': row['url']}
        print(f'{index}: {results}')

        esg_similarities = {
            'System Context Prompt 1': {'Environmental': 0, 'Social': 0, 'Governance': 0},
            'System Context Prompt 2': {'Environmental': 0, 'Social': 0, 'Governance': 0},
            'User Context Prompt 1': {'Environmental': 0, 'Social': 0, 'Governance': 0},
            'User Context Prompt 2': {'Environmental': 0, 'Social': 0, 'Governance': 0},
            'Non-System Context Prompt 1': {'Environmental': 0, 'Social': 0, 'Governance': 0},
            'Non-System Context Prompt 2': {'Environmental': 0, 'Social': 0, 'Governance': 0},
        }

        for i, prompt in enumerate(prompts_1):
            esg_sentence = model.extract_esg_sentence(prompt, verbose=False)
            results[f'ESG Sentences System Prompt {i+1}'] = esg_sentence

            environmental_sentences, social_sentences, governance_sentences = utils.parse_esg_json(esg_sentence)

            if environmental_sentences:
                esg_similarities[f'System Context Prompt {i+1}']['Environmental'] = utils.calculate_pairwise_cosine_similarity_str(environmental_sentences)
                all_embeddings[f'System Context Prompt {i+1}']['Environmental'].extend(utils.encode_arr(environmental_sentences))

            if social_sentences:
                esg_similarities[f'System Context Prompt {i+1}']['Social'] = utils.calculate_pairwise_cosine_similarity_str(social_sentences)
                all_embeddings[f'System Context Prompt {i+1}']['Social'].extend(utils.encode_arr(social_sentences))

            if governance_sentences:
                esg_similarities[f'System Context Prompt {i+1}']['Governance'] = utils.calculate_pairwise_cosine_similarity_str(governance_sentences)
                all_embeddings[f'System Context Prompt {i+1}']['Governance'].extend(utils.encode_arr(governance_sentences))

        for i, prompt in enumerate(prompts_1):
            esg_sentence = model_2.extract_esg_sentence(prompt, verbose=False)
            results[f'ESG Sentences Non-System Prompt {i+1}'] = esg_sentence

            environmental_sentences, social_sentences, governance_sentences = utils.parse_esg_json(esg_sentence)

            if environmental_sentences:
                esg_similarities[f'Non-System Context Prompt {i+1}']['Environmental'] = utils.calculate_pairwise_cosine_similarity_str(environmental_sentences)
                all_embeddings[f'Non-System Context Prompt {i+1}']['Environmental'].extend(utils.encode_arr(environmental_sentences))

            if social_sentences:
                esg_similarities[f'Non-System Context Prompt {i+1}']['Social'] = utils.calculate_pairwise_cosine_similarity_str(social_sentences)
                all_embeddings[f'Non-System Context Prompt {i+1}']['Social'].extend(utils.encode_arr(social_sentences))

            if governance_sentences:
                esg_similarities[f'Non-System Context Prompt {i+1}']['Governance'] = utils.calculate_pairwise_cosine_similarity_str(governance_sentences)
                all_embeddings[f'Non-System Context Prompt {i+1}']['Governance'].extend(utils.encode_arr(governance_sentences))


        for i, prompt in enumerate(prompts_2):
            esg_sentence = model_2.extract_esg_sentence(prompt, verbose=False)
            results[f'ESG Sentences User Prompt {i+1}'] = esg_sentence

            environmental_sentences, social_sentences, governance_sentences = utils.parse_esg_json(esg_sentence)

            if environmental_sentences:
                esg_similarities[f'User Context Prompt {i+1}']['Environmental'] = utils.calculate_pairwise_cosine_similarity_str(environmental_sentences)
                all_embeddings[f'User Context Prompt {i+1}']['Environmental'].extend(utils.encode_arr(environmental_sentences))

            if social_sentences:
                esg_similarities[f'User Context Prompt {i+1}']['Social'] = utils.calculate_pairwise_cosine_similarity_str(social_sentences)
                all_embeddings[f'User Context Prompt {i+1}']['Social'].extend(utils.encode_arr(social_sentences))

            if governance_sentences:
                esg_similarities[f'User Context Prompt {i+1}']['Governance'] = utils.calculate_pairwise_cosine_similarity_str(governance_sentences)
                all_embeddings[f'User Context Prompt {i+1}']['Governance'].extend(utils.encode_arr(governance_sentences))

        results['System Context ESG Cosine Similarity Prompt 1'] = f"E: {esg_similarities['System Context Prompt 1']['Environmental']} S: {esg_similarities['System Context Prompt 1']['Social']} G: {esg_similarities['System Context Prompt 1']['Governance']}"
        results['System Context ESG Cosine Similarity Prompt 2'] = f"E: {esg_similarities['System Context Prompt 2']['Environmental']} S: {esg_similarities['System Context Prompt 2']['Social']} G: {esg_similarities['System Context Prompt 2']['Governance']}"
        results['User Context ESG Cosine Similarity Prompt 1'] = f"E: {esg_similarities['User Context Prompt 1']['Environmental']} S: {esg_similarities['User Context Prompt 1']['Social']} G: {esg_similarities['User Context Prompt 1']['Governance']}"
        results['User Context ESG Cosine Similarity Prompt 2'] = f"E: {esg_similarities['User Context Prompt 2']['Environmental']} S: {esg_similarities['User Context Prompt 2']['Social']} G: {esg_similarities['User Context Prompt 2']['Governance']}"
        results['Non-System Context ESG Cosine Similarity Prompt 1'] = f"E: {esg_similarities['User Context Prompt 1']['Environmental']} S: {esg_similarities['User Context Prompt 1']['Social']} G: {esg_similarities['User Context Prompt 1']['Governance']}"
        results['Non-System Context ESG Cosine Similarity Prompt 2'] = f"E: {esg_similarities['User Context Prompt 2']['Environmental']} S: {esg_similarities['User Context Prompt 2']['Social']} G: {esg_similarities['User Context Prompt 2']['Governance']}"
        
        data.append(results)

    # Create a DataFrame from the data list
    result_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    result_df.to_csv("results/system_context_test_3.csv", index=False)

    # Plot the results
    #plot_cosine_similarities(result_df)

    print(f"""
            Total embeddings:
            System Context Prompt 1:
            E: {len(all_embeddings['System Context Prompt 1']['Environmental'])}
            S: {len(all_embeddings['System Context Prompt 1']['Social'])}
            G: {len(all_embeddings['System Context Prompt 1']['Governance'])}
            System Context Prompt 2:
            E: {len(all_embeddings['System Context Prompt 2']['Environmental'])}
            S: {len(all_embeddings['System Context Prompt 2']['Social'])}
            G: {len(all_embeddings['System Context Prompt 2']['Governance'])}
            User Context Prompt 1:
            E: {len(all_embeddings['User Context Prompt 1']['Environmental'])}
            S: {len(all_embeddings['User Context Prompt 1']['Social'])}
            G: {len(all_embeddings['User Context Prompt 1']['Governance'])}
            User Context Prompt 2:
            E: {len(all_embeddings['User Context Prompt 2']['Environmental'])}
            S: {len(all_embeddings['User Context Prompt 2']['Social'])}
            G: {len(all_embeddings['User Context Prompt 2']['Governance'])}
            Non-System Context Prompt 1:
            E: {len(all_embeddings['Non-System Context Prompt 1']['Environmental'])}
            S: {len(all_embeddings['Non-System Context Prompt 1']['Social'])}
            G: {len(all_embeddings['Non-System Context Prompt 1']['Governance'])}
            Non-System Context Prompt 2:
            E: {len(all_embeddings['Non-System Context Prompt 2']['Environmental'])}
            S: {len(all_embeddings['Non-System Context Prompt 2']['Social'])}
            G: {len(all_embeddings['Non-System Context Prompt 2']['Governance'])}
            
            Overall Cosine similarity
            System Context Prompt 1:
            E: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['System Context Prompt 1']['Environmental'])}
            S: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['System Context Prompt 1']['Social'])}
            G: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['System Context Prompt 1']['Governance'])}
            System Context Prompt 2:
            E: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['System Context Prompt 2']['Environmental'])}
            S: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['System Context Prompt 2']['Social'])}
            G: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['System Context Prompt 2']['Governance'])}
            User Context Prompt 1:
            E: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['User Context Prompt 1']['Environmental'])}
            S: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['User Context Prompt 1']['Social'])}
            G: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['User Context Prompt 1']['Governance'])}
            User Context Prompt 2:
            E: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['User Context Prompt 2']['Environmental'])}
            S: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['User Context Prompt 2']['Social'])}
            G: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['User Context Prompt 2']['Governance'])}
            Non-System Context Prompt 1:
            E: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['Non-System Context Prompt 1']['Environmental'])}
            S: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['Non-System Context Prompt 1']['Social'])}
            G: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['Non-System Context Prompt 1']['Governance'])}
            Non-System Context Prompt 2:
            E: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['Non-System Context Prompt 2']['Environmental'])}
            S: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['Non-System Context Prompt 2']['Social'])}
            G: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings['Non-System Context Prompt 2']['Governance'])}
        """)

def plot_cosine_similarities(df):
    prompts = ['System Context Prompt 1', 'System Context Prompt 2', 'User Context Prompt 1', 'User Context Prompt 2']
    categories = ['Environmental', 'Social', 'Governance']

    for category in categories:
        similarities = [
            df[f'System Context ESG Cosine Similarity Prompt 1'].apply(lambda x: float(x.split()[1 if category == 'Environmental' else 3 if category == 'Social' else 5][:-1]) if x.split()[1 if category == 'Environmental' else 3 if category == 'Social' else 5][:-1] else 0),
            df[f'System Context ESG Cosine Similarity Prompt 2'].apply(lambda x: float(x.split()[1 if category == 'Environmental' else 3 if category == 'Social' else 5][:-1]) if x.split()[1 if category == 'Environmental' else 3 if category == 'Social' else 5][:-1] else 0),
            df[f'User Context ESG Cosine Similarity Prompt 1'].apply(lambda x: float(x.split()[1 if category == 'Environmental' else 3 if category == 'Social' else 5][:-1]) if x.split()[1 if category == 'Environmental' else 3 if category == 'Social' else 5][:-1] else 0),
            df[f'User Context ESG Cosine Similarity Prompt 2'].apply(lambda x: float(x.split()[1 if category == 'Environmental' else 3 if category == 'Social' else 5][:-1]) if x.split()[1 if category == 'Environmental' else 3 if category == 'Social' else 5][:-1] else 0)
        ]
        plt.figure()
        plt.bar(prompts, [similarities[0].mean(), similarities[1].mean(), similarities[2].mean(), similarities[3].mean()])
        plt.xlabel('Prompts')
        plt.ylabel(f'{category} Cosine Similarity')
        plt.title(f'Mean {category} Cosine Similarity Across Prompts')
        plt.ylim(0, 1)
        plt.show()

if __name__ == '__main__':
    main()
