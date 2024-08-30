import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import random
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT

import utils

random.seed(42)

system_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production, HPC.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection (Hacks), Entry Barrier and Accessibility (Global Reach, User Adoptions, Investment), Market Instability (Price Drops and Increases), Illicit Activities, Large Financial Institutions and Crypto Institution
- Governance (G): Decentralized Governance Models (Off-chain and On-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
"""

examples = """
Example 1 (Environmental):
Input: Bitcoin miner Marathon Digital (MARA) has become a multicoin miner to diversify its revenue stream as the recent Bitcoin halving cut profits by 50% and made the industry more competitive. The company has mined 93 million kaspa (KAS) tokens since September 2023, valued at about $15 million, and brought 30 petahash worth of machines online to mine the token, while 30 more will be starting by the third quarter.
Output: Environmental
Justification: While it's true that the overall mining activity will likely continue regardless of MARA's involvement, the focus on energy consumption remains relevant. MARA's decision to expand its mining operations contributes to the cumulative energy demand of the cryptocurrency network, thereby increasing the overall carbon footprint.

Example 2 (Adversarial):
Input: The recent price drop was caused by miner sales, some pressure from German-seized BTC, and, of course, the imminent transfer of Mt. Gox coins expected in early July," Strijers said.
Output: Adversarial
Justification: Although 'miner' can link to mining operation, this sentence doesn't directly contributes to Environmental issue like carbon emission.

Example 3 (Social):
Input: Bitcoin’s price volatility shakes out weak hands and provides opportunities for strategic capital deployment to those with a longer time horizon
Output: Social
Justification: Social concern of short term holder.

Example 4 (Adversarial):
Input: A lower-than-expected figure would suggest a continued inflation decline and potentially boost cryptocurrencies in the coming months
Output: Adversarial
Justification: A surge in crypto investment is a normal market behaviour.

Example 5 (Governance):
Input: Also, CME Group, owner of the Chicago Mercantile Exchange, is close to overtaking Binance as the largest crypto derivatives exchange in the world. (CME's product is a cash-settled futures contract, essentially a side bet on bitcoin's price; no BTC changes hands)
Output: Governance
Justification: CME Group overtaking Binance underscores the evolving landscape of crypto trading (through regulated financial products and more). CME Group as a regulated entity could signal a move towards greater regulatory adherence in the crypto market exchange platform like Binance, potentially increase market stability and investor protection.

Example 6 (Adversarial):
Input: In addition, while some bulls may have been excited that the presidential debate seemed to favor Donald Trump – who has recently come out as notably pro-crypto and pro-bitcoin
Output: Adversarial
Justification: Although this sentence hints at community engagement with public figure like the president candidate might affect the public perception and potential governmental influence on crypto, it doesn't has any tangible impact or actions to be considered an ESG issue.
"""
def create_prompt(title, content):
    return [f"""
          Article Title: {title}
          Article Context: {content}
        
          Task: Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics. 
          Examples: {examples}
          Return your answer in a JSON array format with each identified sentence as a string.

        """,
        f"""
            Article Title: {title}
            Article Context: {content}
            
            Task: Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics. 
            Examples: {examples}

            Let's think step by step.
            Step 1: Identify and explain any Environmental (E) aspects mentioned in the article.
            Environmental Aspects:

            Step 2: Based on Step 1, extract the original sentences from the article that relates to the Environmental Aspects. Return the sentences in a JSON array.
            Environmental Array:

            Step 3: Identify and explain any Social (S) aspects mentioned in the article. 
            Social Aspects:

            Step 4: Based on Step 3, extract the original sentences from the article that relates to the Social Aspects. Return the sentences in a JSON array.
            Social Array:
            
            Step 5: Identify and explain any Governance (G) aspects mentioned in the article.
            Governance Aspects:

            Step 6: Based on Step 5, extract the original sentences from the article that relates to the Governance Aspects. Return the sentences in a JSON array.
            Governance Array:
        """]

def main():
    model = GPT(system_context=system_context)
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    ground_truth_path = '../data/ground_truth.csv'
    ground_truth = utils.read_ground_truth_from_csv(ground_truth_path)

    rows_indices = [i for i in range(0, 22) if i not in [6, 14]]  # Exclude specific articles

    # Initialize a list to store the sentences and their corresponding ESG-related sentences
    data = []
    all_embeddings = []
    for index in rows_indices:
        row = df.iloc[index]
        prompts = create_prompt(row['title'], row['content'])
        results = {'Title': row['title'], 'URL': row['url']}
        print(results)

        for i, prompt in enumerate(prompts):
            esg_sentence = model.extract_esg_sentence(prompt, verbose=False)
            results[f'ESG Sentences Prompt {i+1}'] = esg_sentence
            esg_sentence_list = utils.parse_esg_json(esg_sentence)
        
            tp, fp, fn, all_iou, best_iou = utils.evaluate_extracted_sentences(esg_sentence_list, ground_truth[index + 1])
            print(f'tp: {tp}, fp: {fp}, fn: {fn}, all_iou: {all_iou:.4f}, best_iou: {best_iou:.4f}')
            all_embeddings.extend(utils.encode_arr(esg_sentence_list))

            # Store metrics for this agent count
            results[f'prompt {i+1} TP'] = tp
            results[f'prompt {i+1} FP'] = fp
            results[f'prompt {i+1} FN'] = fn
            results[f'prompt {i+1} All IOU'] = all_iou
            results[f'prompt {i+1} Best IOU'] = best_iou
        
        data.append(results)


    #Create a DataFrame from the data list
    result_df = pd.DataFrame(data)

    #Save the DataFrame to a CSV file
    result_df.to_csv("results/few_shots_test.csv", index=False)
    print(f'Overall Cosine Similarity: {utils.calculate_pairwise_cosine_similarity_embedding(all_embeddings)}')

if __name__ == '__main__':
    main()