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
import random


random.seed(42)
def generate_initial_responses(model, num_agents, title, content):
    extracted_sentences = set()
    initial_prompt = f"""
                    Article Title: {title}
                    Article Context: {content}

                    Task:
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
                """
    for _ in range(num_agents):
        response = model.extract_esg_sentence(initial_prompt, temperature=random.random(), verbose=False)
        
        all_sentences = utils.parse_esg_json(response)
        
        # Process sentences: Upper-case the first letter and add to the set
        processed_sentences = {sentence.capitalize() for sentence in all_sentences}
        extracted_sentences.update(processed_sentences)

    print(f'EXTRACTED: {extracted_sentences} \n\n')
    return list(extracted_sentences)

def critique_responses(model, num_agents, title, content, sentences):
    tasks = [
        "Given the article context and extracted ESG-relevant sentences, go through each sentence and critique their relevancy according to the key themes.",
        "Given the article context and extracted ESG-relevant sentences, evaluate if the sentence capture the full scope of the ESG issue, or is it missing key details?",
        "Given the article context and extracted ESG-relevant sentences, analyze if there are underlying implications or connections to the key themes that may not be immediately obvious but could be relevant upon closer examination."
    ]
    
    critiques = {}
    
    for i in range(num_agents):
        # Assign a task to each agent based on its index
        task = tasks[i % len(tasks)]
        
        critique_prompt = f"""
                            Article Title: {title}
                            Article Context: {content}
                            Extracted ESG Sentences: {sentences}
                            Key themes: 
                            - Environmental (E): Energy Consumption, Carbon Emissions, Resource Management, Renewable Energy Usage, Electronic Waste Production, HPC.
                            - Social (S): Labor Practice, Community Engagement and Inclusion, Security and User Protection (Hacks), Entry Barrier and Accessibility (Global Reach, User Adoptions, Investment), Market Instability (Price Drops and Increases), Illicit Activities, Large Financial Institutions and Crypto Institution
                            - Governance (G): Decentralized Governance Models (Off-chain and On-chain), Business Ethics and Transparency, Regulatory Compliance, Executive Compensation and Incentives, Tax Evasion, Geographical Differences and Regulatory Challenges
                            
                            Task: {task}
                            """
        
        for idx, sentence in enumerate(sentences, 1):
            critique_prompt += f"""
                            Sentence {idx}: "{sentence}"
                            Critique:
                            """
        
        critique = model.extract_esg_sentence(critique_prompt, temperature=random.random(), top_p=0.4, verbose=False)
        critique_list = re.findall(r'Critique:\s*(.*?)(?=\s*Sentence \d+:|$)', critique, re.DOTALL)
        #print(f'critique_list: {critique_list}')
        # Store each critique in the dictionary with the sentence_index as the key
        for sentence_index, critique_text in enumerate(critique_list, 1):
            # Initialize the list if it's the first time this key is being used
            if sentence_index not in critiques:
                critiques[sentence_index] = []
            
            # Append the critique to the list for this key
            critiques[sentence_index].append(critique_text.strip())

    #print(f'Critiques: {critiques} \n\n')
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
        refined_response = model.extract_esg_sentence(refinement_prompt, temperature=random.random(), verbose=False)
        #print(refined_response)
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
        
        #print(f'DECISION: {matches}')
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
    model = GPT(system_context=utils.system_context)

    num_agents = 3
    
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)
    ground_truth_path = '../data/ground_truth.csv'
    ground_truth = utils.read_ground_truth_from_csv(ground_truth_path)

    rows_indices = [i for i in range(0, 22) if i not in [6, 14]]
    

    # Initialize a DataFrame to store the sentences and their corresponding ESG-related sentences for all indices
    all_logs = []

    for index in rows_indices:
        row = df.iloc[index]

        results = {'Title': row['title'], 'URL': row['url']}
        print(results)

        print("Initial Response Round \n")
        extracted_sentences = generate_initial_responses(model, num_agents, row['title'], row['content'])
        
        # Generate critiques
        print("\nCritique Round \n")
        critiques = critique_responses(model, num_agents, row['title'], row['content'], extracted_sentences)
        
        # Refine responses based on critiques
        print("\nRefined Responses\n")
        matches, refined_sentences = refine_responses(model, num_agents, row['title'], row['content'], extracted_sentences, critiques)
        
        tp, fp, fn, all_iou, best_iou = utils.evaluate_extracted_sentences(refined_sentences, ground_truth[index + 1])
        print(f'tp: {tp}, fp: {fp}, fn: {fn}, all_iou: {all_iou:.4f}, best_iou: {best_iou:.4f}')
        
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
        results['TP'] = tp 
        results['FP'] = fp
        results['FN'] = fn 
        results['All IoU'] = all_iou
        results['Best IoU'] = best_iou
        
        all_logs.append(results)
        
    # Save the combined DataFrame to a CSV file
    all_logs_df = pd.DataFrame(all_logs)
    all_logs_df.to_csv("results/debate_test_2.csv", index=False)

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