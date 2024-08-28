from sentence_transformers import SentenceTransformer, util
import numpy as np

import json
import re

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
    Extract the JSON formatted array/list from GPT's output

    Example output:
    ### Step 2: Identify and explain any Social (S) aspects mentioned in the context.

    **Social Aspects:**
    The article touches on social aspects indirectly through the discussion of the U.S. Presidential Elections and the potential impact of political outcomes on the cryptocurrency market. The social aspect here is the public sentiment and market predictions regarding the election.

    **Social Array:**
    ```json
    [
        "Traders on the decentralized predictions platform Polymarket have already chosen a winner in the 2024 U.S. Presidential Elections, and it's not incumbent Joe Biden.",
        "A Polymarket contract asking who would win the election showed Republican candidate Donald Trump as the clear favorite, with a 57% chance of winning versus 35% for Biden.",
        "Meanwhile, another contract showed Biden winning the popular vote, which merely represents the proportion of votes cast for each candidate and carries no electoral weight."
    ]
    ```

    Parameters:
    output (str): Output retrieved from the model. Expected **Social/Environmental/Governance Array:**, followed by ```json ```.
    keyword (str): Social/Environmental/Governance

    Returns:
    json_array (list): list of extracted sentences 
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

# def parse_esg_json(json_data):
#     """
#     Parse the ESG JSON data and return the Environmental, Social, and Governance arrays.

#     Parameters:
#     json_data (str): A JSON string containing the ESG data, potentially surrounded by ```json and ```.

#     Returns:
#     tuple: A tuple containing three lists - Environmental, Social, and Governance arrays.
#     """
#     # Strip the ```json and ``` surrounding the actual JSON content
#     if json_data.startswith("```json") and json_data.endswith("```"):
#         json_data = json_data[7:-3]
#         # Replace invalid escape sequences
#         json_data = json_data.replace('\\', '\\\\')

#         try:
#             # Load the JSON data
#             data = json.loads(json_data)
            
#             # environmental_array = data.get("Environmental", [])
#             # social_array = data.get("Social", [])
#             # governance_array = data.get("Governance", [])
            
#             return data

#         except json.JSONDecodeError as e:
#             print(f"JSONDecodeError: {e}")
#         return [],[],[]
#     else: 
#         environmental_array = extract_json_array(json_data, "Environmental")
#         social_array = extract_json_array(json_data, "Social")
#         governance_array = extract_json_array(json_data, "Governance")

#         return environmental_array + social_array + governance_array

def parse_esg_json(json_data):
    """
    Parse the JSON data and combine all JSON arrays into a single list.

    Parameters:
    json_data (str): A string containing JSON arrays potentially surrounded by ```json and ```.

    Returns:
    list: A single list containing all elements from all JSON arrays found in the text.
    """
    # Regex to find all JSON arrays enclosed in ```json ... ```
    matches = re.findall(r'```json\s*(.*?)\s*```', json_data, re.DOTALL)
    if not matches: 
        print(f'================NO LIST: {matches}')
    
    combined_list = []
    
    for json_array_str in matches:
        try:
            # Parse the JSON array
            json_array = json.loads(json_array_str)
            
            # Check if the parsed JSON is a list (array)
            if isinstance(json_array, list):
                # Extend the combined list with elements from the current JSON array
                combined_list.extend(json_array)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
    
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
    #print(f'extracted: {extracted_sentences}')
    #print(f'ground truth: {ground_truth_sentences}')

    # Initialize counts for evaluation
    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truth_sentences)

    # Token sets for each sentence
    ground_truth_sets = [preprocess_sentence(gt) for gt in ground_truth_sentences]
    extracted_sets = [preprocess_sentence(ext) for ext in extracted_sentences]

    # Initialize sets for calculating overall token coverage
    all_gt_tokens = set()  # All tokens in ground truth sentences
    matched_extracted_tokens = set()  # Tokens covered by matched extracted sentences
    all_extracted_tokens = set()  # All tokens in extracted sentences

    # Fill all_gt_tokens with all unique tokens from ground truth sentences
    for gt_set in ground_truth_sets:
        all_gt_tokens.update(gt_set)

    # Fill all_extracted_tokens with all unique tokens from extracted sentences
    for ext_set in extracted_sets:
        all_extracted_tokens.update(ext_set)

    # Track the best match for each ground truth sentence
    best_matches = [None] * len(ground_truth_sets)  # Store the best matching extracted set for each ground truth
    best_iou_scores = [0] * len(ground_truth_sets)  # Store the best IoU score for each ground truth

    # Match extracted sentences with ground truth sentences
    for ext_set in extracted_sets:
        match_found = False  # Flag to indicate if a match is found
        for idx, gt_set in enumerate(ground_truth_sets):
            iou = calculate_iou(ext_set, gt_set)
            # print("========================================")
            # print(f'ext_set: {ext_set}')
            # print(f'gt_set: {gt_set}')
            # print(f'iou: {iou}')
            # print("========================================")
            if iou >= iou_threshold and iou > best_iou_scores[idx]:
                best_iou_scores[idx] = iou
                best_matches[idx] = ext_set
                match_found = True

        if match_found:
            true_positives += 1
            false_negatives -= 1
        else:
            false_positives += 1

    # Add tokens from the best matching extracted sentences to matched_extracted_tokens
    for matched_set in best_matches:
        if matched_set:
            matched_extracted_tokens.update(matched_set)

    # Calculate traditional IoU based on all tokens
    intersection_tokens = matched_extracted_tokens.intersection(all_gt_tokens)
    union_tokens = all_gt_tokens.union(all_extracted_tokens)
    overall_iou = len(intersection_tokens) / len(union_tokens) if union_tokens else 0

    # print(f'INTERSECT: {intersection_tokens}')
    # print(f'LEN OF GT: {all_gt_tokens}')
    # Calculate percentage of ground truth tokens covered
    token_coverage_percentage = len(intersection_tokens) / len(all_gt_tokens) if all_gt_tokens else 0

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return true_positives, false_positives, false_negatives, overall_iou, token_coverage_percentage, precision, recall


