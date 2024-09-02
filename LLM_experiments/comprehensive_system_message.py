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

def generate_prompts(title, content, system_context=None):
    """Generate prompts based on the title and content, optionally including a system context."""
    
    # Prepend system context if provided, otherwise use an empty string
    context_prefix = f"{system_context}\n" if system_context else ""
    
    return [
        f"""
        {context_prefix}Article Title: {title}
        Article Context: {content}

        Task: Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics. 
        Return your answer in a JSON array format with each identified sentence as a string.
        """,
        f"""
        {context_prefix}Article Title: {title}
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
    ]

def extract_and_evaluate(model, prompts, ground_truth):
    """Extract sentences using the model and evaluate them."""
    results = {}
    all_sentence_embeddings = []

    for i, prompt in enumerate(prompts):
        esg_sentences = model.extract_esg_sentence(prompt, verbose=False)
        results[f'Prompt {i + 1} ESG Sentences'] = esg_sentences

        # Parse and encode sentences
        esg_sentence_list = utils.parse_esg_json(esg_sentences)
        embeddings = utils.encode_arr(esg_sentence_list)
        
        all_sentence_embeddings.append(embeddings)
            

        # Evaluate extracted sentences
        tp, fp, fn, overall_iou, best_iou = utils.evaluate_extracted_sentences(esg_sentence_list, ground_truth)
        #print(f'tp: {tp}, fp: {fp}, fn: {fn}, all_iou: {overall_iou:.4f}, best_iou:{best_iou:.4f}')
        
        # Store evaluation metrics
        results[f'Prompt {i + 1} TP'] = tp
        results[f'Prompt {i + 1} FP'] = fp
        results[f'Prompt {i + 1} FN'] = fn
        results[f'Prompt {i + 1} All IOU'] = overall_iou
        results[f'Prompt {i + 1} Best IOU'] = best_iou

    return results, all_sentence_embeddings

def read_ground_truth_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    ground_truth_by_article = df.groupby('id')['sentence'].apply(list).to_dict()
    return ground_truth_by_article

def main():
    model_system = GPT(system_context=utils.system_context)
    model_non_system = GPT()

    # Load your data using pandas
    data_file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(data_file_path)
    ground_truth_path = '../data/ground_truth.csv'
    ground_truth = read_ground_truth_from_csv(ground_truth_path)

    rows_indices = [i for i in range(0, 22) if i not in [6, 14]]  # Exclude specific articles
    #rows_indices = [0,1]
    all_logs = []

    # Initialize lists to store embeddings for each of the six experiments
    embeddings_dict = {
        'System ZS': [],
        'System CoT': [],
        'Non-System ZS': [],
        'Non-System CoT': [],
        'User-System ZS': [],
        'User-System CoT': []
    }

    for index in rows_indices:
        row = df.iloc[index]
        results = {'Title': row['title'], 'URL': row['url']}
        print(f'Processing Article {index}: {results["Title"]}')

        # Generate prompts for different experiments
        prompts_system = generate_prompts(row['title'], row['content'], utils.system_context)
        prompts_non_system = generate_prompts(row['title'], row['content'])

        # System context experiment
        system_results, system_embeddings = extract_and_evaluate(model_system, prompts_system, ground_truth[index + 1])
        embeddings_dict['System ZS'].extend(system_embeddings[0])
        embeddings_dict['System CoT'].extend(system_embeddings[1])
        for key, value in system_results.items():
            results[f'System {key}'] = value

        # Non-system context experiment
        non_system_results, non_system_embeddings = extract_and_evaluate(model_non_system, prompts_non_system, ground_truth[index + 1])
        embeddings_dict['Non-System ZS'].extend(non_system_embeddings[0])
        embeddings_dict['Non-System CoT'].extend(non_system_embeddings[1])
        for key, value in non_system_results.items():
            results[f'Non-System {key}'] = value

        # User prompt context experiment (non-system prompt with user context)
        user_results, user_embeddings = extract_and_evaluate(model_non_system, prompts_system, ground_truth[index + 1])
        embeddings_dict['User-System ZS'].extend(user_embeddings[0])
        embeddings_dict['User-System CoT'].extend(user_embeddings[1])
        for key, value in user_results.items():
            results[f'User-System {key}'] = value
            
        all_logs.append(results)

    # Create a DataFrame from the data list
    result_df = pd.DataFrame(all_logs)
    result_df.to_csv("results/updated_system_context_test_new.csv", index=False)

    # Calculate overall cosine similarity for all six experiments
    #print(embeddings_dict)
    for experiment_name, embeddings in embeddings_dict.items():
        if len(embeddings) > 1:  # Ensure there are at least two embeddings for cosine similarity
            cosine_similarity = utils.calculate_pairwise_cosine_similarity_embedding(embeddings)
            print(f"{experiment_name} Overall Cosine Similarity: {cosine_similarity}")

if __name__ == "__main__":
    main()
