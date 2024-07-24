from sentence_transformers import SentenceTransformer, util
import numpy as np

import json
import re

# Load pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_similarity(text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()

def calculate_pairwise_cosine_similarity(responses):
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
    return np.mean(similarities)

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
