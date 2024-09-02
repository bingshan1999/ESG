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

def generate_initial_responses(model, num_agents, title, content):
    extracted_sentences = set()
    initial_prompt = f"""
                    Article Title: {title}
                    Article Context: {content}

                    Task: 
                    Let's think step by step.
                    Step 1: Identify and explain any Environmental (E) aspects mentioned in the article.
                    Environmental Aspects:

                    Step 2: Based on Step 1, extract the original sentences from the article that relate to the Environmental Aspects. Return the sentences in a JSON array.
                    Environmental Array:

                    Step 3: Identify and explain any Social (S) aspects mentioned in the article.
                    Social Aspects:

                    Step 4: Based on Step 3, extract the original sentences from the article that relate to the Social Aspects. Return the sentences in a JSON array.
                    Social Array:

                    Step 5: Identify and explain any Governance (G) aspects mentioned in the article.
                    Governance Aspects:

                    Step 6: Based on Step 5, extract the original sentences from the article that relate to the Governance Aspects. Return the sentences in a JSON array.
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

def read_ground_truth_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    ground_truth_by_article = df.groupby('id')['sentence'].apply(list).to_dict()
    return ground_truth_by_article

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
    model = GPT(system_context=utils.system_context)
    num_agents_initial = 3
    num_agents_critique = 3
    num_iterations = 5

    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)
    ground_truth_path = '../data/ground_truth.csv'
    ground_truth = read_ground_truth_from_csv(ground_truth_path)
    all_logs = []
    rows_indices = [i for i in range(0, 22) if i not in [6, 14]]  # Exclude specific articles
    #rows_indices = [0,1]

    # Initialize a nested dictionary to store metrics for each row and iteration
    metrics_over_iterations = {index: {iteration: {'TP': 0, 'FP': 0, 'FN': 0, 'All IoU': 0, 'Best IoU': 0, 'IoU Improvement': 0, 'IoU Decrease': 0}
                                       for iteration in range(1, num_iterations + 1)}
                               for index in rows_indices}

    for index in rows_indices:
        row = df.iloc[index]
        results = {'Title': row['title'], 'URL': row['url']}
        print(results)

        ######################### First iteration
        extracted_sentences = generate_initial_responses(model, num_agents_initial, row['title'], row['content'])
        critiques = critique_responses(model, num_agents_critique, row['title'], row['content'], extracted_sentences)   
        matches, refined_sentences = refine_responses(model, num_agents_critique, row['title'], row['content'], extracted_sentences, critiques)

        # Calculate metrics for the first iteration
        tp, fp, fn, all_iou, best_iou = utils.evaluate_extracted_sentences(refined_sentences, ground_truth[index + 1])
        print(f'tp: {tp}, fp: {fp}, fn: {fn}, all_iou: {all_iou:.4f}, best_iou:{best_iou:.4f}')


        # Store the metrics
        metrics_over_iterations[index][0] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'All IoU': all_iou,
            'Best IoU': best_iou,
            'IoU Improvement': 0,
            'IoU Decrease': 0
        }
        
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
        results['TP'] = tp 
        results['FP'] = fn 
        results['FN'] = fn
        results['All IoU'] = all_iou 
        results['Best IoU'] = best_iou

        ############################ NEXT ITERATIONS
        for iteration in range(1, num_iterations + 1):
            print(f"==============ITERATION: {iteration}==============")
            new_article = filter_content(row['content'], refined_sentences)
            #print(f'NEW ARTICLE: {new_article}')
            
            new_extracted_sentences = generate_initial_responses(model, num_agents_initial, row['title'], new_article) 
            new_critiques = critique_responses(model, num_agents_critique, row['title'], row['content'], new_extracted_sentences) # pass the original article for full context
            matches, new_refined_sentences = refine_responses(model, num_agents_critique, row['title'], row['content'], new_extracted_sentences, new_critiques)
            
            final_ans = refined_sentences + new_refined_sentences

            new_tp, new_fp, new_fn, new_all_iou, new_best_iou = utils.evaluate_extracted_sentences(final_ans, ground_truth[index + 1])
            print(f'tp: {tp}, fp: {fp}, fn: {fn}, all_iou: {all_iou:.4f}, best_iou:{best_iou:.4f}')

            iou_improvement = new_all_iou - metrics_over_iterations[index][iteration - 1]['All IoU']
 
            # Store the metrics for the current iteration
            metrics_over_iterations[index][iteration]['TP'] = new_tp
            metrics_over_iterations[index][iteration]['FP'] = new_fp
            metrics_over_iterations[index][iteration]['FN'] = new_fn
            metrics_over_iterations[index][iteration]['All IoU'] = new_all_iou
            metrics_over_iterations[index][iteration]['Best IoU'] = new_best_iou
            metrics_over_iterations[index][iteration]['IoU Improvement'] = 1 if iou_improvement > 0 else 0
            metrics_over_iterations[index][iteration]['IoU Decrease'] = 1 if iou_improvement < 0 else 0

            # Store the extracted sentences, critiques, and refined sentences
            results[f'Iteration {iteration} Extracted Sentences'] = "\n".join(new_extracted_sentences)
            critiques_str = ""
            for sentence_idx, agent_critiques in new_critiques.items():
                critiques_str += f"Sentence {sentence_idx}:\n"
                for i, critique in enumerate(agent_critiques):
                    critiques_str += f"  Agent {i+1}:\n    " + "\n    ".join(critique.splitlines()) + "\n"
            results[f'Iteration {iteration} Critiques'] = critiques_str
            results[f'Iteration {iteration} New Sentences'] = "\n".join(final_ans)
            results[f'Iteration {iteration} TP'] = new_tp 
            results[f'Iteration {iteration} FP'] = new_fp 
            results[f'Iteration {iteration} FN'] = new_fn 
            results[f'Iteration {iteration} All IoU'] = new_all_iou
            results[f'Iteration {iteration} Best IoU'] = new_best_iou
            print(f'{len(new_refined_sentences)} Additions with {iou_improvement:.4f} IOU changes')
            refined_sentences = final_ans
        
        all_logs.append(results)
        
     # Save the combined DataFrame to a CSV file
    all_logs_df = pd.DataFrame(all_logs)
    all_logs_df.to_csv("results/feedback_debate_test_COT.csv", index=False)
    #print(metrics_over_iterations)
    # Plot metrics
    plot_metrics_over_iterations(metrics_over_iterations)
    plot_iou_changes(metrics_over_iterations)

def plot_metrics_over_iterations(metrics_over_iterations):
    """Plot Average IoU, Precision, Recall, and Best IoU over iterations."""

    plt.figure(figsize=(12, 8))

    # Determine the number of iterations
    iterations = range(1, max(next(iter(metrics_over_iterations.values())).keys()) + 1)  # Correctly determine the range of iterations

    avg_iou_arr = []
    avg_best_iou_arr = []
    precisions = []
    recalls = []

    for iteration in iterations:
        total_tp = total_fp = total_fn = 0
        all_iou_sum = best_iou_sum = 0
        num_rows = len(metrics_over_iterations)  # The number of rows being processed

        for index in metrics_over_iterations:
            total_tp += metrics_over_iterations[index][iteration]['TP']
            total_fp += metrics_over_iterations[index][iteration]['FP']
            total_fn += metrics_over_iterations[index][iteration]['FN']
            all_iou_sum += metrics_over_iterations[index][iteration]['All IoU']
            best_iou_sum += metrics_over_iterations[index][iteration]['Best IoU']

        # Calculate average precision, recall, All IOU, and Best IOU
        precision_value = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall_value = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        avg_all_iou = all_iou_sum / num_rows
        avg_best_iou = best_iou_sum / num_rows

        precisions.append(precision_value)
        recalls.append(recall_value)
        avg_iou_arr.append(avg_all_iou)
        avg_best_iou_arr.append(avg_best_iou)

    # Plot Average IoU, Precision, Recall, and Best IoU
    plt.plot(iterations, avg_iou_arr, marker='o', linestyle='-', color='b', label='Average All IoU')
    plt.plot(iterations, avg_best_iou_arr, marker='o', linestyle='-', color='c', label='Average Best IoU')
    plt.plot(iterations, precisions, marker='o', linestyle='-', color='g', label='Average Precision')
    plt.plot(iterations, recalls, marker='o', linestyle='-', color='r', label='Average Recall')

    # Set labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.title('Average IoU, Precision, Recall, and Best IoU Over Iterations')
    plt.legend(loc='best')
    plt.xticks(iterations)
    plt.grid(True)
    plt.show()

def plot_iou_changes(metrics_over_iterations):
    """Plot the number of rows with IoU increases and decreases across iterations."""

    # Define the iterations we have data for
    iterations = range(1, max(next(iter(metrics_over_iterations.values())).keys()) + 1)

    # Count the number of rows with IoU improvements and decreases for each iteration
    improvements = [sum(metrics_over_iterations[row][i]['IoU Improvement'] for row in metrics_over_iterations) for i in iterations]
    decreases = [sum(metrics_over_iterations[row][i]['IoU Decrease'] for row in metrics_over_iterations) for i in iterations]

    # Plotting the data
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    plt.bar([i - bar_width / 2 for i in iterations], improvements, width=bar_width, label='IoU Increase', color='green', align='center')
    plt.bar([i + bar_width / 2 for i in iterations], decreases, width=bar_width, label='IoU Decrease', color='red', align='center')

    # Setting labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Number of Rows')
    plt.title('Number of Rows with IoU Increase and Decrease Across Iterations')
    plt.xticks(iterations)
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    main()