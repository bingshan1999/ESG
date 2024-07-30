import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
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
    avg_cosine_similarity = utils.calculate_pairwise_cosine_similarity_str(combined_sentences)
    
    # Calculate intersection
    intersection_set = set(sentences[0])
    for sent_set in sentences[1:]:
        intersection_set &= set(sent_set)
    
    # Calculate the average length of sentence sets to get the intersection percentage
    avg_length = sum(len(sent) for sent in sentences) / len(sentences)
    intersection_percentage = len(intersection_set) / avg_length if avg_length > 0 else 0
    
    counts = []
    for i, sent in enumerate(sentences):
        count = len(sent)
        counts.append(count)
        print(f"Response {i+1} {keyword} - Number of Sentences: {count}")
        
    return avg_cosine_similarity, intersection_percentage, counts

def plot_metrics(all_metrics, title):
    num_rounds = all_metrics['Round'].nunique()
    num_agents = all_metrics['Agent'].nunique()

    fig, axes = plt.subplots(num_rounds, num_agents + 1, figsize=(20, 5 * num_rounds), sharey='row')
    
    rounds = all_metrics['Round'].unique()
    agents = all_metrics['Agent'].unique()
    categories = ['Environmental', 'Social', 'Governance']
    category_prefixes = {'Environmental': 'E', 'Social': 'S', 'Governance': 'G'}

    for round_idx, round_num in enumerate(rounds):
        round_data = all_metrics[all_metrics['Round'] == round_num]
        
        for agent_idx, agent in enumerate(agents):
            agent_data = round_data[round_data['Agent'] == agent]
            
            counts = []
            for category in categories:
                prefix = category_prefixes[category]
                initial_data = agent_data[agent_data['Type'] == 'Initial Response'][f'{prefix} count']
                refined_data = agent_data[agent_data['Type'] == 'Refined Response'][f'{prefix} count']
                initial_count = initial_data.iloc[0] if not initial_data.empty else 0
                refined_count = refined_data.iloc[0] if not refined_data.empty else 0
                counts.append((category, initial_count, refined_count))
            
            count_df = pd.DataFrame(counts, columns=['Category', 'Initial Count', 'Refined Count'])
            melted_counts = count_df.melt(id_vars='Category', value_vars=['Initial Count', 'Refined Count'], var_name='Type', value_name='Count')
            
            sns.barplot(x='Category', y='Count', hue='Type', data=melted_counts, ax=axes[round_idx, agent_idx])
            axes[round_idx, agent_idx].set_title(f'Round {round_num} - Agent {agent}')
            axes[round_idx, agent_idx].set_ylabel('Count')
            axes[round_idx, agent_idx].set_xlabel('Category')

        intersection_metrics = []
        for category in categories:
            prefix = category_prefixes[category]
            initial_cosine_data = round_data[(round_data['Type'] == 'Initial Response')][f'{prefix} cosine similarity']
            refined_cosine_data = round_data[(round_data['Type'] == 'Refined Response')][f'{prefix} cosine similarity']
            initial_intersection_data = round_data[(round_data['Type'] == 'Initial Response')][f'{prefix} intersection']
            refined_intersection_data = round_data[(round_data['Type'] == 'Refined Response')][f'{prefix} intersection']

            initial_cosine_similarity = initial_cosine_data.iloc[0] if not initial_cosine_data.empty else 0
            refined_cosine_similarity = refined_cosine_data.iloc[0] if not refined_cosine_data.empty else 0
            initial_intersection_percentage = initial_intersection_data.iloc[0] / 100 if not initial_intersection_data.empty else 0  # Scale intersection percentage
            refined_intersection_percentage = refined_intersection_data.iloc[0] / 100 if not refined_intersection_data.empty else 0  # Scale intersection percentage

            intersection_metrics.append((category, 'Initial', initial_cosine_similarity, initial_intersection_percentage))
            intersection_metrics.append((category, 'Refined', refined_cosine_similarity, refined_intersection_percentage))

        intersection_df = pd.DataFrame(intersection_metrics, columns=['Category', 'Type', 'Cosine Similarity', 'Intersection Percentage'])
        #melted_intersection = intersection_df.melt(id_vars=['Category', 'Type'], value_vars=['Cosine Similarity', 'Intersection Percentage'], var_name='Metric', value_name='Value')

        # Debug print to check the content of melted_intersection
        #print(f"Round {round_num} - Intersection Data:\n", melted_intersection)
        
        # Line plot for cosine similarity and intersection percentage
        for category in categories:
            category_data = intersection_df[intersection_df['Category'] == category]
            sns.lineplot(x='Type', y='Cosine Similarity', data=category_data, marker='o', label=f'{category} Cosine Similarity', ax=axes[round_idx, num_agents])
            sns.lineplot(x='Type', y='Intersection Percentage', data=category_data, marker='x', label=f'{category} Intersection %', ax=axes[round_idx, num_agents])

        axes[round_idx, num_agents].set_title(f'Round {round_num} - Metrics')
        axes[round_idx, num_agents].set_ylabel('Cosine Similarity / Intersection Percentage')
        axes[round_idx, num_agents].set_xlabel('Category')
        axes[round_idx, num_agents].legend(loc='upper right')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def generate_initial_responses(model, prompt, num_agents):
    responses = []
    for _ in range(num_agents):
        response = model.extract_esg_sentence(prompt, temperature=0.7, verbose=False)
        responses.append(response)
    return responses

def critique_responses(model, prompt, initial_responses):
    critiques = []
    for i in range(len(initial_responses)):
        # The next agent's response in a circular manner
        next_agent_response = initial_responses[(i + 1) % len(initial_responses)]  
        critique_prompt = f"""
                            {prompt}
                            
                            Previous Responses from Other Agents: 
                            {next_agent_response}
                            
                            Critique the above responses by addressing the following points:
                            1. Identify any factual errors.
                            2. Point out logical inconsistencies.
                            3. Suggest areas of improvement.
                            """
        critique = model.extract_esg_sentence(critique_prompt, temperature=0.7, verbose=False)
        critiques.append(critique)
    return critiques

def refine_responses(model, prompt, initial_responses, critiques):
    refined_responses = []
    for i in range(len(initial_responses)):
        previous_response = initial_responses[i]
        critique_text = critiques[(i - 1) % len(initial_responses)]
        refinement_prompt = f"""
                            {prompt}

                            Your Previous Response:
                            {previous_response}

                            Critiques of Your Previous Response:
                            {critique_text}

                            Based on the critiques and feedback provided, refine your response by:
                            1. Correcting any factual errors identified.
                            2. Addressing logical inconsistencies.
                            3. Incorporating any additional information provided by other agents.
                            4. Strengthening your arguments and reasoning.
                            """
        refined_response = model.extract_esg_sentence(refinement_prompt, temperature=0.7, verbose=False)
        refined_responses.append(refined_response)
    return refined_responses

def debate_process(model, prompt, title, url, num_agents=2, max_rounds=3, convergence_threshold=0.95):
    # Initialize a DataFrame to store the logs
    log_columns = ['Title', 'URL', 'Round', 'Agent', 'Type', 'Content', 
                   'E cosine similarity', 'E intersection', 'E count',
                   'S cosine similarity', 'S intersection', 'S count',
                   'G cosine similarity', 'G intersection', 'G count']
    logs = pd.DataFrame(columns=log_columns)

    print("Initial Responses\n")
    initial_responses = generate_initial_responses(model, prompt, num_agents)
    
    for round_num in range(max_rounds):
        round_label = round_num + 1
        print(f"\n{'='*20} ROUND {round_label} {'='*20}\n")
        
        # Calculate similarity and intersection for initial responses
        e_initial_cosine_similarity, e_initial_intersection_percentage, e_initial_counts = calculate_combined_similarity(initial_responses, "Environmental")
        s_initial_cosine_similarity, s_initial_intersection_percentage, s_initial_counts = calculate_combined_similarity(initial_responses, "Social")
        g_initial_cosine_similarity, g_initial_intersection_percentage, g_initial_counts = calculate_combined_similarity(initial_responses, "Governance")
        
        for agent_num, response in enumerate(initial_responses):
            agent_label = agent_num + 1
            log_entry = pd.DataFrame([[title, url, round_label, agent_label, 'Initial Response', response, 
                                       e_initial_cosine_similarity, e_initial_intersection_percentage, e_initial_counts[agent_num],
                                       s_initial_cosine_similarity, s_initial_intersection_percentage, s_initial_counts[agent_num],
                                       g_initial_cosine_similarity, g_initial_intersection_percentage, g_initial_counts[agent_num]]], 
                                      columns=log_columns)
            log_entry = log_entry.fillna('')
            logs = pd.concat([logs, log_entry], ignore_index=True)
        
        # Generate critiques
        print("\nCritiques\n")
        critiques = critique_responses(model, prompt, initial_responses)
    
        for agent_num, critique in enumerate(critiques):
            agent_label = agent_num + 1
            log_entry = pd.DataFrame([[title, url, round_label, agent_label, 'Critique', critique, 
                                       None, None, None, None, None, None, None, None, None]], 
                                      columns=log_columns)
            log_entry = log_entry.fillna('')
            logs = pd.concat([logs, log_entry], ignore_index=True)
        
        # Refine responses based on critiques
        print("\nRefined Responses\n")
        refined_responses = refine_responses(model, prompt, initial_responses, critiques)
        
        # Calculate similarity, intersection, and counts for refined responses
        e_refined_cosine_similarity, e_refined_intersection_percentage, e_refined_counts = calculate_combined_similarity(refined_responses, "Environmental")
        s_refined_cosine_similarity, s_refined_intersection_percentage, s_refined_counts = calculate_combined_similarity(refined_responses, "Social")
        g_refined_cosine_similarity, g_refined_intersection_percentage, g_refined_counts = calculate_combined_similarity(refined_responses, "Governance")


        for agent_num, refined_response in enumerate(refined_responses):
            agent_label = agent_num + 1
            log_entry = pd.DataFrame([[title, url, round_label, agent_label, 'Refined Response', refined_response, 
                                       e_refined_cosine_similarity, e_refined_intersection_percentage, e_refined_counts[agent_num],
                                       s_refined_cosine_similarity, s_refined_intersection_percentage, s_refined_counts[agent_num],
                                       g_refined_cosine_similarity, g_refined_intersection_percentage, g_refined_counts[agent_num]]], 
                                      columns=log_columns)
            log_entry = log_entry.fillna('')
            logs = pd.concat([logs, log_entry], ignore_index=True)
        
        # Print refined similarities and intersections
        print(f"""
            Initial response:
                E cosine similarity: {e_initial_cosine_similarity}
                E intersection percentage: {e_initial_intersection_percentage:.4%}
                S cosine similarity: {s_initial_cosine_similarity}
                S intersection percentage: {s_initial_intersection_percentage:.4%}
                G cosine similarity: {g_initial_cosine_similarity}
                G intersection percentage: {g_initial_intersection_percentage:.4%}
            Refined response:
                E cosine similarity: {e_refined_cosine_similarity}
                E intersection percentage: {e_refined_intersection_percentage:.4%}
                S cosine similarity: {s_refined_cosine_similarity}
                S intersection percentage: {s_refined_intersection_percentage:.4%}
                G cosine similarity: {g_refined_cosine_similarity}
                G intersection percentage: {g_refined_intersection_percentage:.4%}
            """)
        
        # Check for convergence
        if (e_refined_cosine_similarity >= convergence_threshold and e_refined_intersection_percentage > 0 and
            s_refined_cosine_similarity >= convergence_threshold and s_refined_intersection_percentage > 0 and
            g_refined_cosine_similarity >= convergence_threshold and g_refined_intersection_percentage > 0):
            print("Convergence achieved.")
            break
        
        initial_responses = refined_responses
    
    return logs

def main():
    model = GPT()
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    rows_indices = [0, 20]
    max_round = 3
    
    # Initialize a DataFrame to store the sentences and their corresponding ESG-related sentences for all indices
    all_logs = pd.DataFrame()

    for index in rows_indices:
        row = df.iloc[index]
        prompt = create_prompt(row['title'], row['content'])
        results = {'Title': row['title'], 'URL': row['url']}
        print(results)

        logs = debate_process(model, prompt, row['title'], row['url'], max_rounds=max_round)
        all_logs = pd.concat([all_logs, logs], ignore_index=True)
        
        plot_metrics(all_logs[all_logs['Title'] == row['title']], row['title'])
    
    # Save the combined DataFrame to a CSV file
    all_logs.to_csv("results/debate_combined.csv", index=False)

if __name__ == '__main__':
    main()