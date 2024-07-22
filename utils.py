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

def extract_json_array(output, keyword):
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
