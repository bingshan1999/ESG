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
import random
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT
from collections import defaultdict
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
        #all_sentences = E + S + G  # Merge all sentences into one list
        
        # Process sentences: Upper-case the first letter and add to the set
        processed_sentences = {sentence.strip('\'"').capitalize() for sentence in all_sentences}

        extracted_sentences.update(processed_sentences)

    print(f'EXTRACTED: {extracted_sentences} \n\n')
    return list(extracted_sentences)

def generate_next_responses(model, num_agents, title, content):
    extracted_sentences = set()
    initial_prompt = f"""
                        Article Title: {title}
                        Article Context: {content}

                        Task: Carefully read all sentences in the article and identify any sentences that relate to Environmental, Social, and Governance (ESG) issues. 
                        Focus on finding sentences that may have been overlooked due to their indirect references to ESG topics or subtle implications.

                        Consider sentences that:
                        - Provide additional information or context about ESG issues that were not captured before.
                        - Mention ESG topics indirectly or through nuanced language that may have been missed in a more straightforward extraction.

                        Return the new sentences in a JSON array format. If no new sentences are found, return an empty JSON array.
                        """

    for _ in range(num_agents):
        response = model.extract_esg_sentence(initial_prompt, temperature=random.random(), verbose=False)
        print(f'=======================response: {response}')
        all_sentences = utils.parse_esg_json(response)
        #all_sentences = E + S + G  # Merge all sentences into one list
        
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

def filter_content(content, previous_sentences, iou_threshold=0.5):
    """
    Remove sentences from the article content that have a high IoU with previously extracted sentences.

    Parameters:
    - content (str): The full text of the article.
    - previous_sentences (list): A list of previously extracted sentences.
    - iou_threshold (float): The IoU threshold above which sentences are considered similar.

    Returns:
    - str: The article content with sentences similar to previously extracted ones removed.
    """
    # Tokenize previous sentences and preprocess
    previous_sets = [utils.preprocess_sentence(sentence) for sentence in previous_sentences]
    
    # Use nltk to split the content into sentences for more accurate tokenization
    sentences = sent_tokenize(content)
    
    filtered_sentences = []

    for sentence in sentences:
        sentence_set = utils.preprocess_sentence(sentence)
        is_similar = False
        
        # Check IoU with each previously extracted sentence
        for prev_set in previous_sets:
            iou = utils.calculate_iou(sentence_set, prev_set)
            if iou >= iou_threshold:
                is_similar = True
                break
        
        if not is_similar:
            filtered_sentences.append(sentence)

    # Join the filtered sentences back into a single string
    filtered_content = " ".join(filtered_sentences)
    return filtered_content

def main():
    model = GPT(system_context=system_context)
    #num_iterations = 3
    num_agents = range(3,7)
    num_criqitues = 3
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)
    ground_truth_path = '../data/ground_truth.csv'
    ground_truth = utils.read_ground_truth_from_csv(ground_truth_path)

    rows_indices = [i for i in range(0, 22) if i not in [6, 14]]  # Exclude specific articles
    #rows_indices = [0]  # For testing with a single row

    # Store metrics for different agent counts
    metrics_per_agent = {num_agent: {'Precision': [], 'Recall': [], 'All IOU': [], 'Best IOU': []} for num_agent in num_agents}

    for num_agent in num_agents:
        for index in rows_indices:
            row = df.iloc[index]
            results = {'Title': row['title'], 'URL': row['url']}
            print(results)

            # First iteration
            extracted_sentences = generate_initial_responses(model, num_agent, row['title'], row['content'])
            critiques = critique_responses(model, num_criqitues, row['title'], row['content'], extracted_sentences)
            matches, refined_sentences = refine_responses(model, num_criqitues, row['title'], row['content'], extracted_sentences, critiques)

            # Calculate metrics for the first iteration
            tp, fp, fn, all_iou, best_iou = utils.evaluate_extracted_sentences(refined_sentences, ground_truth[index + 1])
            print(f'Number of agent: {num_agent}, tp: {tp}, fp: {fp}, fn: {fn}, all_iou: {all_iou:.4f}, best_iou: {best_iou:.4f}')
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Store metrics for this agent count
            metrics_per_agent[num_agent]['Precision'].append(precision)
            metrics_per_agent[num_agent]['Recall'].append(recall)
            metrics_per_agent[num_agent]['All IOU'].append(all_iou)
            metrics_per_agent[num_agent]['Best IOU'].append(best_iou)

    # Compute average metrics for each agent count
    avg_metrics_per_agent = {
        num_agent: {
            'Precision': sum(metrics_per_agent[num_agent]['Precision']) / len(metrics_per_agent[num_agent]['Precision']),
            'Recall': sum(metrics_per_agent[num_agent]['Recall']) / len(metrics_per_agent[num_agent]['Recall']),
            'All IOU': sum(metrics_per_agent[num_agent]['All IOU']) / len(metrics_per_agent[num_agent]['All IOU']),
            'Best IOU': sum(metrics_per_agent[num_agent]['Best IOU']) / len(metrics_per_agent[num_agent]['Best IOU'])
        }
        for num_agent in metrics_per_agent
    }

    # Plot the metrics over different agent counts
    plot_metrics_vs_agents(avg_metrics_per_agent)

    # Save the combined DataFrame to a CSV file
    #all_logs_df = pd.DataFrame(all_logs)
    #all_logs_df.to_csv("results/feedback_debate_test_2.csv", index=False)

def plot_metrics_vs_agents(avg_metrics_per_agent):
    """Plot Precision, Recall, All IOU, and Best IOU vs. Number of Agents."""

    agents = list(avg_metrics_per_agent.keys())
    precision = [avg_metrics_per_agent[agent]['Precision'] for agent in agents]
    recall = [avg_metrics_per_agent[agent]['Recall'] for agent in agents]
    all_iou = [avg_metrics_per_agent[agent]['All IOU'] for agent in agents]
    best_iou = [avg_metrics_per_agent[agent]['Best IOU'] for agent in agents]

    plt.figure(figsize=(10, 6))
    plt.plot(agents, precision, marker='o', linestyle='-', color='b', label='Precision')
    plt.plot(agents, recall, marker='o', linestyle='-', color='g', label='Recall')
    plt.plot(agents, all_iou, marker='o', linestyle='-', color='r', label='All IOU')
    plt.plot(agents, best_iou, marker='o', linestyle='-', color='c', label='Best IOU')

    plt.xlabel('Number of Agents')
    plt.ylabel('Percentage')
    plt.title('Metrics vs. Number of Agents')
    plt.legend(loc='best')
    plt.xticks(agents)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()