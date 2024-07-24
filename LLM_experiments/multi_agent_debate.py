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

def create_prompt(title, content):
    return f"""
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
def calculate_combined_similarity(responses, keyword):
    sentences = [utils.extract_json_array(response, keyword) for response in responses]
    combined_sentences = [' '.join(sent) for sent in sentences]
    
    # Calculate pairwise cosine similarities using embeddings
    avg_cosine_similarity = utils.calculate_pairwise_cosine_similarity(combined_sentences)
    
    # Calculate intersection
    intersection_set = set(sentences[0])
    for sent_set in sentences[1:]:
        intersection_set &= set(sent_set)
    intersection_count = len(intersection_set)
    
    return avg_cosine_similarity, intersection_count

def generate_initial_responses(model, prompt, num_agents):
    responses = []
    for _ in range(num_agents):
        response = model.extract_esg_sentence(prompt, temperature=0.7, verbose=False)
        responses.append(response)
    return responses

def critique_responses(model, prompt, initial_responses):
    critiques = []
    for i in range(len(initial_responses)):
        combined_responses = initial_responses[:i] + initial_responses[i+1:]
        combined_text = " ".join(combined_responses)
        critique_prompt = f"{prompt}\n\nPrevious Responses from Other Agents:\n{combined_text}\n\nCritique the above responses."
        critique = model.extract_esg_sentence(critique_prompt, verbose=False)
        critiques.append(critique)
    return critiques

def refine_responses(model, prompt, initial_responses, critiques):
    refined_responses = []
    for i in range(len(initial_responses)):
        combined_responses = initial_responses[:i] + initial_responses[i+1:]
        critique_text = critiques[i]
        refinement_prompt = f"{prompt}\n\nPrevious Responses from Other Agents:\n{' '.join(combined_responses)}\n\nCritiques of Your Previous Response:\n{critique_text}\n\nBased on the critiques and feedback provided, refine your response."
        refined_response = model.extract_esg_sentence(refinement_prompt, verbose=False)
        refined_responses.append(refined_response)
    return refined_responses

def debate_process(model, prompt, title, url, num_agents=2, max_rounds=3, convergence_threshold=0.95):
    # Initialize a DataFrame to store the logs
    log_columns = ['Title', 'URL', 'Round', 'Agent', 'Type', 'Content']
    logs = pd.DataFrame(columns=log_columns)

    initial_responses = generate_initial_responses(model, prompt, num_agents)
    #previous_similarity = 0
    
    # Calculate similarity and intersection for initial responses
    e_initial_cosine_similarity, e_initial_intersection_count = calculate_combined_similarity(initial_responses, "Environmental")
    s_initial_cosine_similarity, s_initial_intersection_count = calculate_combined_similarity(initial_responses, "Social")
    g_initial_cosine_similarity, g_initial_intersection_count = calculate_combined_similarity(initial_responses, "Governance")
    
    for round_num in range(max_rounds):
        round_label = f"ROUND {round_num + 1}"
        print(f"\n{'='*20} {round_label} {'='*20}\n")
        
        # Print and log initial responses
        print("Initial Responses\n")
        for agent_num, response in enumerate(initial_responses):
            agent_label = f"Agent {agent_num + 1}"
            #print(f"{agent_label} Response:\n{response}\n")
            log_entry = pd.DataFrame([[title, url, round_label, agent_label, 'Initial Response', response]], columns=log_columns)
            logs = pd.concat([logs, log_entry], ignore_index=True)
        
        # Generate critiques
        critiques = critique_responses(model, prompt, initial_responses)
        
        # Print and log critiques
        print("\nCritiques\n")
        for agent_num, critique in enumerate(critiques):
            agent_label = f"Agent {agent_num + 1}"
            #print(f"{agent_label} Critique:\n{critique}\n")
            log_entry = pd.DataFrame([[title, url, round_label, agent_label, 'Critique', critique]], columns=log_columns)
            logs = pd.concat([logs, log_entry], ignore_index=True)
        
        # Refine responses based on critiques
        refined_responses = refine_responses(model, prompt, initial_responses, critiques)
        
        # Print and log refined responses
        print("\nRefined Responses\n")
        for agent_num, refined_response in enumerate(refined_responses):
            agent_label = f"Agent {agent_num + 1}"
            #print(f"{agent_label} Refined Response:\n{refined_response}\n")
            log_entry = pd.DataFrame([[title, url, round_label, agent_label, 'Refined Response', refined_response]], columns=log_columns)
            logs = pd.concat([logs, log_entry], ignore_index=True)
        
        # Calculate similarity between refined responses
        e_refined_cosine_similarity, e_refined_intersection_count = calculate_combined_similarity(refined_responses, "Environmental")
        s_refined_cosine_similarity, s_refined_intersection_count = calculate_combined_similarity(refined_responses, "Social")
        g_refined_cosine_similarity, g_refined_intersection_count = calculate_combined_similarity(refined_responses, "Governance")

        # Calculate similarity between refined responses
        # similarity = calculate_similarity(refined_responses)
        # print(f"Round {round_num + 1}, Similarity: {similarity}")
        print(f"""
        Initial response:
            E cosine similarity: {e_initial_cosine_similarity}
            E intersection: {e_initial_intersection_count}
            S cosine similarity: {s_initial_cosine_similarity}
            S intersection: {s_initial_intersection_count}
            G cosine similarity: {g_initial_cosine_similarity}
            G intersection: {g_initial_intersection_count}
        Refined response:
            E cosine similarity: {e_refined_cosine_similarity}
            E intersection: {e_refined_intersection_count}
            S cosine similarity: {s_refined_cosine_similarity}
            S intersection: {s_refined_intersection_count}
            G cosine similarity: {g_refined_cosine_similarity}
            G intersection: {g_refined_intersection_count}
        """)

        # Check for convergence
        if (e_refined_cosine_similarity >= convergence_threshold and e_refined_intersection_count > 0 and
            s_refined_cosine_similarity >= convergence_threshold and s_refined_intersection_count > 0 and
            g_refined_cosine_similarity >= convergence_threshold and g_refined_intersection_count > 0):
            print("Convergence achieved.")
            break
        
        #previous_similarity = similarity
        initial_responses = refined_responses

    return logs

model = GPT()
# Load your data using pandas
file_path = '../data/cleaned_coindesk_btc.csv'
df = pd.read_csv(file_path)

rows_indices = [0, 20]

# Initialize a list to store the sentences and their corresponding ESG-related sentences
all_logs = pd.DataFrame()

for index in rows_indices:
    row = df.iloc[index]
    prompt = create_prompt(row['title'], row['content'])
    results = {'Title': row['title'], 'URL': row['url']}
    print(results)
    
    logs = debate_process(model, prompt, row['title'], row['url'])
    all_logs = pd.concat([all_logs, logs], ignore_index=True)

# Save the combined DataFrame to a CSV file
all_logs.to_csv("results/debate_combined.csv", index=False)