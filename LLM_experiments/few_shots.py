import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import random
import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT

import utils

random.seed(42)

examples_list = [
"""
Example 1:
Sentence: "As traditional financial institutions start to dominate the cryptocurrency landscape, one might wonder if this could be a game-changer for the industry."
ESG analysis: This sentence poses a rhetorical question about the influence of traditional finance on the cryptocurrency sector but does not provide any concrete information or evidence about an ESG impact. It lacks specificity and fails to address any tangible issues that would concern ESG criteria.
""",

"""
Example 2:
Sentence: "At the same time, another poll indicated that Biden would win the popular vote, which simply reflects the total number of individual votes and has no bearing on the electoral college outcome."
ESG analysis: This sentence focuses solely on election results and the distribution of the popular vote, which is unrelated to cryptocurrencies or any ESG issues. While different presidential candidates may have varying stances on crypto, this sentence does not mention any direct connection between the election outcome and ESG concerns. 
""",

"""
Example 3: 
Sentence: "ETH's disappointing first quarter price action might be a correction within a larger bull trend, following a rise from December 2021 lows to a new all-time high above $40,000 in mid-March."
ESG analysis: The reference to significant price fluctuations, including a new all-time high, highlights market volatility. This could affect investor confidence and financial stability, which are relevant to the social and governance aspects of ESG.
""",

"""
Example 4:
Sentence: "BTC price increases as decentralization became prominent"
ESG analysis: The sentence mention a price surge but without further context, this is considered as a common fluctuations and market behaviour, which doesn't constitue to market instability.   
"""

]

def create_prompt(title, content, examples):
    return [f"""
          Article Title: {title}
          Article Context: {content}
        
          Task: Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics. 
          Examples: 
          {examples}
          Return your answer in a JSON array format with each identified sentence as a string.

        """,
        f"""
            Article Title: {title}
            Article Context: {content}
            
            Task: Identify any sentences from the article that might involve ESG (Environmental, Social, Governance) topics. 
            Examples: 
            {examples}

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
    model = GPT(system_context=utils.system_context)
    # Load your data using pandas
    file_path = '../data/cleaned_coindesk_btc.csv'
    df = pd.read_csv(file_path)

    ground_truth_path = '../data/ground_truth.csv'
    ground_truth = utils.read_ground_truth_from_csv(ground_truth_path)

    rows_indices = [i for i in range(0, 22) if i not in [6, 14]]  # Exclude specific articles
    #rows_indices = [0,1]
    
    # Initialize a list to store the sentences and their corresponding ESG-related sentences
    data = []
    all_embeddings = []
     # Initialize a nested dictionary to store metrics for each row and iteration
    metrics_over_examples = {num_examples: {index: {'TP': 0, 'FP': 0, 'FN': 0, 'All IOU': 0, 'Best IOU': 0}
                                            for index in rows_indices}
                             for num_examples in range(1, len(examples_list) + 1)}
    
    # Range for number of examples to use in prompts
    num_examples_range = range(1, len(examples_list) + 1)  # Corrected to use a range object

    for num_examples in num_examples_range:  # Changed to iterate over the range
        print(f'Current number of examples: {num_examples}')
        for index in rows_indices:
            row = df.iloc[index]
            examples = "\n".join(examples_list[0:num_examples])  # Corrected to use num_examples as an integer
            #print(f'examples: {examples}')
            prompts = create_prompt(row['title'], row['content'], examples)
            results = {'Title': row['title'], 'URL': row['url']}
            print(results)

            for i, prompt in enumerate(prompts):
                esg_sentence = model.extract_esg_sentence(prompt, verbose=False)
                results[f'ESG Sentences Prompt {i+1}'] = esg_sentence
                esg_sentence_list = utils.parse_esg_json(esg_sentence)

                tp, fp, fn, all_iou, best_iou = utils.evaluate_extracted_sentences(esg_sentence_list, ground_truth[index + 1])
                print(f'tp: {tp}, fp: {fp}, fn: {fn}, all_iou: {all_iou:.4f}, best_iou: {best_iou:.4f}')
                all_embeddings.extend(utils.encode_arr(esg_sentence_list))

                # Store the metrics
                metrics_over_examples[num_examples][index] = {  # Corrected to use num_examples
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'All IOU': all_iou,
                    'Best IOU': best_iou
                }

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
    plot_metrics_vs_examples(metrics_over_examples)

def plot_metrics_vs_examples(metrics_over_examples):
    """Plot Precision, Recall, All IOU, and Best IOU vs. Number of Examples."""
    examples = list(metrics_over_examples.keys())

    # Initialize lists to store aggregated metrics
    precision = []
    recall = []
    all_iou = []
    best_iou = []

    for example in examples:
        total_tp = total_fp = total_fn = 0
        all_iou_sum = best_iou_sum = 0

        for index in metrics_over_examples[example]:
            total_tp += metrics_over_examples[example][index]['TP']
            total_fp += metrics_over_examples[example][index]['FP']
            total_fn += metrics_over_examples[example][index]['FN']
            all_iou_sum += metrics_over_examples[example][index]['All IOU']
            best_iou_sum += metrics_over_examples[example][index]['Best IOU']

        # Calculate precision and recall
        precision_value = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall_value = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        precision.append(precision_value)
        recall.append(recall_value)

        # Calculate average All IOU and Best IOU
        avg_all_iou = all_iou_sum / len(metrics_over_examples[example]) if len(metrics_over_examples[example]) > 0 else 0
        avg_best_iou = best_iou_sum / len(metrics_over_examples[example]) if len(metrics_over_examples[example]) > 0 else 0
        all_iou.append(avg_all_iou)
        best_iou.append(avg_best_iou)

    plt.figure(figsize=(10, 6))
    plt.plot(examples, precision, marker='o', linestyle='-', color='b', label='Precision')
    plt.plot(examples, recall, marker='o', linestyle='-', color='g', label='Recall')
    plt.plot(examples, all_iou, marker='o', linestyle='-', color='r', label='All IOU')
    plt.plot(examples, best_iou, marker='o', linestyle='-', color='c', label='Best IOU')

    plt.xlabel('Number of Examples')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Number of Examples')
    plt.legend(loc='best')
    plt.xticks(examples)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()