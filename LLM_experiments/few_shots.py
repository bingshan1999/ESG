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
system_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection, Entry Barrier and Accessibility, Market Instability, Illicit Activities, Influence of Large Financial Institutions and Crypto Institution
- Governance (G): Decentralized Governance Models (off-chain and on-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
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
          Examples: {examples}

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
            Examples: {examples}
            
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
    model = GPT(system_context=system_context)
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = range(0,21)
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
    result_df.to_csv("results/few_shots_test.csv", index=False)

if __name__ == '__main__':
    main()