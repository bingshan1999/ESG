import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import re 
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT
from collections import defaultdict
import utils

system_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given an article, you will be asked to extract ESG issues from it. 
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection, Entry Barrier and Accessibility, Market Instability, Illicit Activities, Influence of Large Financial Institutions and Crypto Institution
- Governance (G): Decentralized Governance Models (off-chain and on-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
"""

critique_context = """
You are critical reviewer with deep expertise in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Given a list of ESG related sentences, you should evaluate each of the provided sentences, justifying if they are relevant to ESG concerns.
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection, Entry Barrier and Accessibility, Market Instability, Illicit Activities, Influence of Large Financial Institutions and Crypto Institution
- Governance (G): Decentralized Governance Models (off-chain and on-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
"""

refinement_context = """
You are an expert in Environmental, Social, and Governance (ESG) topics, specifically within the cryptocurrency space. 
Your task is to review a list of ESG-related sentences along with their respective critiques. Based on the critiques, decide whether each sentence should be eliminated from the list.
Here are the key ESG issues that are particularly relevant in the context of cryptocurrencies:

- Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production.
- Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection, Entry Barrier and Accessibility, Market Instability, Illicit Activities, Influence of Large Financial Institutions and Crypto Institution
- Governance (G): Decentralized Governance Models (off-chain and on-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
"""

def generate_initial_responses(model, num_agents, title, content):
    extracted_sentences = set()
    initial_prompt = f"""
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
                """
    for _ in range(num_agents):
        response = model.extract_esg_sentence(initial_prompt, temperature=0.7, verbose=False)
        
        E, S, G = utils.parse_esg_json(response)
        all_sentences = E + S + G  # Merge all sentences into one list
        
        # Process sentences: Upper-case the first letter and add to the set
        processed_sentences = {sentence.capitalize() for sentence in all_sentences}
        extracted_sentences.update(processed_sentences)

    print(f'EXTRACTED: {extracted_sentences} \n\n')
    return list(extracted_sentences)

def critique_responses(model, num_agents, title, content, sentences):
    critiques = {}
    critique_prompt = f"""
                        Article Title: {title}
                        Article Context: {content}
                        Extracted ESG Sentences: {sentences}
                        
                        Task: Given the article context and extracted ESG-relevant sentences, go through each sentence and critique their relevancy.
                        """
    for idx, sentence in enumerate(sentences, 1):
        critique_prompt += f"""
                        Sentence {idx}: "{sentence}"
                        Critique:
                        """
    for i in range(num_agents):
        critique = model.extract_esg_sentence(critique_prompt, temperature=0.7, verbose=False)
 
        critique_list = re.findall(r'Critique:\s*(.*?)\s*(?=(Sentence \d+:|$))', critique, re.DOTALL)
        #print(critique_list)
        # Store each critique in the dictionary with the sentence_index as the key
        for sentence_index, critique_tuple in enumerate(critique_list, 1):
            critique_text = critique_tuple[0]
            # Initialize the list if it's the first time this key is being used
            if sentence_index not in critiques:
                critiques[sentence_index] = []
            
            # Append the critique to the list for this key
            critiques[sentence_index].append(critique_text.strip())

    #print(f'CRITQUE PROMPT: {critique_prompt} \n\n')
    print(f'Critiques: {critiques} \n\n')
    return critiques

def refine_responses(model, num_agents, title, content, sentences, critiques):
    refinement_prompt = f"""
                        Article Title: {title}
                        Article Context: {content}
                        Extracted ESG Sentences: {sentences}
                        
                        Task: Given the article context and extracted ESG-relevant sentences, review the critiques and vote if the sentences should be eliminated from the list.
                        """
    
    for idx, sentence in enumerate(sentences, 1):
        refinement_prompt += f"""
                        Sentence {idx}: "{sentence}"
                        """
        
        # Add critiques for this sentence from all agents
        if idx in critiques:
            for agent_num, critique in enumerate(critiques[idx], 1):
                refinement_prompt += f"""
                        Agent {agent_num} Critique: {critique}
                        """
        
        # Add the decision section
        refinement_prompt += f"""
                        Decision (Include/Exclude):
                        """
    decision_counts = defaultdict(int)
    decision_matches = []
    for i in range(num_agents):
        refined_response = model.extract_esg_sentence(refinement_prompt, temperature=0.7, verbose=False)
        print(refined_response)
        # First regex pattern to match the default format
        default_decision_pattern = r"\**Decision\**:\s*\**\s*(Include|Exclude)\**"

        # Second regex pattern to match the alternative format
        alternative_decision_pattern = r"\**\s*(Include|Exclude)\**"

        # Try the first regex pattern
        matches = re.findall(default_decision_pattern, refined_response, re.DOTALL)

        # If no decisions found with the first regex, try the second regex
        if not matches:
            matches = re.findall(alternative_decision_pattern, refined_response, re.DOTALL)
        
        decision_matches.append(matches)
        
        print(f'DECISION: {matches}')
        # Increment or decrement the count based on the decision
        for idx, decision in enumerate(matches):
            if decision == "Include":
                decision_counts[idx] += 1

    # Apply the final decisions directly to the sentences list
    sentences = [
        sentence for idx, sentence in enumerate(sentences) 
        if decision_counts[idx] > num_agents // 2
    ]

    #print(f'REFINEMENT PROMPT: {refinement_prompt} \n\n')
    print(f'FINAL SENTENCES: {sentences} \n\n')
    return decision_matches, sentences

def main():
    response_model = GPT(system_context=system_context)
    critique_model = GPT(system_context=critique_context)
    refinement_model = GPT(system_context=refinement_context)
    num_agents = 3
    
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = range(0,21)

    # Initialize a DataFrame to store the sentences and their corresponding ESG-related sentences for all indices
    all_logs = []

    for index in rows_indices:
        row = df.iloc[index]

        results = {'Title': row['title'], 'URL': row['url']}
        print(results)

        print("Initial Response Round \n")
        extracted_sentences = generate_initial_responses(response_model, num_agents, row['title'], row['content'])
        
        # Generate critiques
        print("\nCritique Round \n")
        critiques = critique_responses(critique_model, num_agents, row['title'], row['content'], extracted_sentences)
        
        # Refine responses based on critiques
        print("\nRefined Responses\n")
        matches, refined_sentences = refine_responses(refinement_model, num_agents, row['title'], row['content'], extracted_sentences, critiques)

        # Store the extracted sentences, critiques, and refined sentences
        results['Extracted Sentences'] = "\n".join(extracted_sentences)
        critiques_str = ""
        for sentence_idx, agent_critiques in critiques.items():
            critiques_str += f"Sentence {sentence_idx}:\n"
            for i, critique in enumerate(agent_critiques):
                critiques_str += f"  Agent {i+1}:\n    " + "\n    ".join(critique.splitlines()) + "\n"
        results['Critiques'] = critiques_str
        results['Refined Sentences'] = "\n".join(refined_sentences)
        results['Matches'] = "\n\n".join([",".join(sublist) for sublist in matches])
        # Calculate the number of removals
        initial_set = set(extracted_sentences)
        refined_set = set(refined_sentences)
        removed_sentences = initial_set - refined_set
        number_of_removals = len(removed_sentences)
        results['Number of Removals'] = number_of_removals
        
        all_logs.append(results)
        
    # Save the combined DataFrame to a CSV file
    all_logs_df = pd.DataFrame(all_logs)
    all_logs_df.to_csv("results/debate_test_matches.csv", index=False)

if __name__ == '__main__':
    main()


# refined_response = "**Decision:** Include - This sentence is very relevant to Governance (G), discussing the regulatory landscape and challenges cryptocurrencies face with traditional finance's involvement."
# default_decision_pattern = r"\**Decision\**:\s*\**\s*(Include|Exclude)\**"


# # Second regex pattern to match the alternative format
# alternative_decision_pattern = r"\d+\.\s\*\*Sentence \d+\*\*:\s*(Include|Exclude)"

# # Try the first regex pattern
# matches = re.findall(default_decision_pattern, refined_response, re.DOTALL)

# print(matches)
# # If no decisions found with the first regex, try the second regex
# if not matches:
#     matches = re.findall(alternative_decision_pattern, refined_response, re.DOTALL)

# print(f'DECISION: {matches}')