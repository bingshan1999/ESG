from sentence_transformers import SentenceTransformer, util
import numpy as np

import json
import re
import pandas as pd

# Load pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def encode_arr(sentences_arr):
    """
    Encode the entire array into vector embeddings

    Parameters:
    sentences_arr: List of strings

    Returns:
    embeddings
    """
    combined_sentences = [' '.join(sent) for sent in sentences_arr]
    return model.encode(combined_sentences, convert_to_tensor=True)

def calculate_similarity(text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()

def calculate_pairwise_cosine_similarity_embedding(responses):
    """
    Calculate the mean of pairwise cosine similarities among embeddings.

    When there are more than two agents, this function computes the pairwise 
    cosine similarity between all pairs of responses and returns the mean similarity.

    Parameters:
    responses: List of embeddings

    Returns:
    float: The mean of pairwise cosine similarities.
    """
    similarities = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            similarity =  util.pytorch_cos_sim(responses[i], responses[j]).item()
            similarities.append(similarity)
    return np.mean(similarities) if similarities else 0

def calculate_pairwise_cosine_similarity_str(responses):
    """
    Calculate the mean of pairwise cosine similarities among responses.

    When there are more than two agents, this function computes the pairwise 
    cosine similarity between all pairs of responses and returns the mean similarity.

    Parameters:
    responses (list of str): List of responses from different agents. (Extracted E,S,G list from JSON)

    Returns:
    float: The mean of pairwise cosine similarities.
    """
    similarities = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            similarity = calculate_similarity(responses[i], responses[j])
            similarities.append(similarity)
    return np.mean(similarities) if similarities else 0

def extract_json_array(output, keyword):
    """
    
    """
    # Create a pattern to find the specific JSON array section based on the keyword
    pattern = re.compile(r'\*\*' + re.escape(keyword) + r' Array:\*\*\s*```json\s*(\[.*?\])\s*```', re.DOTALL)
    match = pattern.search(output)
    if match:
        json_str = match.group(1)
        # Parse the JSON string into a Python list
        try:
            json_array = json.loads(json_str)
            return json_array
        except json.JSONDecodeError:
            return []
    else:
        return []
    
def lists_to_intersection(*lists):
    if not lists:
        return set()
    
    # Convert the first list to a set
    intersection_set = set(lists[0])
    
    # Iterate over the remaining lists, updating the intersection set
    for lst in lists[1:]:
        intersection_set &= set(lst)
    
    return intersection_set


def parse_esg_json(json_data):
    """
    Parse the JSON data and combine all JSON arrays into a single list.

    Parameters:
    json_data (str): A string containing JSON arrays potentially surrounded by ```json and ```.

    Returns:
    list: A single list containing all elements from all JSON arrays found in the text.
    """
    #json_data = json_data.replace('\\', '')
    #json_data = json_data.replace('\"\"', '\"')
    # Regex to find all JSON arrays enclosed in ```json ... ```
    matches = re.findall(r'```json\s*(.*?)\s*```', json_data, re.DOTALL)
    if not matches: 
        print(f'================NO LIST: {matches}')
    
    combined_list = []
    
    for json_array_str in matches:
        try:
            json_array_str = re.sub(r'\\\'', "'", json_array_str)
            json_array = json.loads(json_array_str)

            # Check if the parsed JSON is a list (array)
            if isinstance(json_array, list):
                # Extend the combined list with elements from the current JSON array
                combined_list.extend(json_array)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"original: {json_data}")
    
    return combined_list

def preprocess_sentence(sentence):
    """Lowercase and tokenize a sentence into a set of words without removing stop words."""
    return set(sentence.lower().split())

def calculate_iou(set1, set2):
    """Calculate Intersection over Union (IoU) for two sets."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    iou = len(intersection) / len(union) if union else 0
    return iou

def evaluate_extracted_sentences(extracted_sentences, ground_truth_sentences, iou_threshold=0.5):
    """Evaluate extracted sentences using IoU against ground truth sentences."""

    # Initialize counts for evaluation
    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truth_sentences)

    # Token sets for each sentence
    ground_truth_sets = [preprocess_sentence(gt) for gt in ground_truth_sentences]
    extracted_sets = [preprocess_sentence(ext) for ext in extracted_sentences]

    # Initialize sets for calculating IoU
    all_gt_tokens = {token for gt_set in ground_truth_sets for token in gt_set}  # All tokens in ground truth sentences
    all_extracted_tokens = {token for ext_set in extracted_sets for token in ext_set}  # All tokens in extracted sentences
    matched_extracted_tokens = set()  # Tokens covered by matched extracted sentences
    matched_gt_tokens = set()  # Tokens in ground truth that have been matched

    # Track the best match for each ground truth sentence
    best_matches = [None] * len(ground_truth_sets)  # Store the best matching extracted set for each ground truth
    best_iou_scores = [0] * len(ground_truth_sets)  # Store the best IoU score for each ground truth

    # Match extracted sentences with ground truth sentences
    for ext_set in extracted_sets:
        for idx, gt_set in enumerate(ground_truth_sets):
            iou = calculate_iou(ext_set, gt_set)

            if iou >= iou_threshold and iou > best_iou_scores[idx]:
                best_iou_scores[idx] = iou
                best_matches[idx] = ext_set

    # Determine true positives, false positives, and false negatives
    matched_indices = set()

    for idx, best_match in enumerate(best_matches):
        if best_match is not None:
            true_positives += 1
            false_negatives -= 1
            matched_indices.add(idx)
            matched_gt_tokens.update(ground_truth_sets[idx])
            matched_extracted_tokens.update(best_match)
        else:
            # If no match is found, still add the GT tokens
            matched_gt_tokens.update(ground_truth_sets[idx])

    false_positives = len(extracted_sets) - true_positives

    # Calculate "Best Match IoU" based on matched tokens only
    intersection_tokens_best_match = matched_extracted_tokens.intersection(matched_gt_tokens)
    union_tokens_best_match = matched_gt_tokens.union(matched_extracted_tokens)
    best_match_iou = len(intersection_tokens_best_match) / len(union_tokens_best_match) if union_tokens_best_match else 0

    # Calculate "All IoU" based on all tokens
    intersection_tokens_all = all_extracted_tokens.intersection(all_gt_tokens)
    union_tokens_all = all_gt_tokens.union(all_extracted_tokens)
    all_iou = len(intersection_tokens_all) / len(union_tokens_all) if union_tokens_all else 0

    return true_positives, false_positives, false_negatives, all_iou, best_match_iou

def read_ground_truth_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    ground_truth_by_article = df.groupby('id')['sentence'].apply(list).to_dict()
    return ground_truth_by_article