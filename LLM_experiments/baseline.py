import pandas as pd
from transformers import pipeline
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load your data using pandas
file_path = '../data/cleaned_coindesk_btc.csv'
df = pd.read_csv(file_path)

# Specify the row indices you want to subset
rows_indices = range(0, 21)

# Create a subset of the DataFrame using the specified row indices
subset_df = df.iloc[rows_indices].copy()

# Initialize the ESG classifier pipeline with device parameter
esg_classifier = pipeline("text-classification", model="nbroad/ESG-BERT", device=device)
#esg_classifier = pipeline("text-classification", model="yiyanghkust/finbert-esg", device=device)

# Function to classify each sentence in an article
def classify_sentences(article_text):
    sentences = sent_tokenize(article_text)
    classifications = []
    for sentence in sentences:
        result = esg_classifier(sentence)
        classifications.append(result)
    return sentences, classifications

# Create a detailed DataFrame for sentence classifications
detailed_data = []

# Apply the classification to each sentence in the subset DataFrame
for index, row in subset_df.iterrows():
    sentences, classifications = classify_sentences(row['content'])
    for sentence, classification in zip(sentences, classifications):
        if classification[0]['score'] > 0.85 :  # and classification[0]['label'] != 'None'
            detailed_data.append({
                'Title': row['title'],
                'URL': row['url'],
                'Sentence': sentence,
                'Label': classification[0]['label'],
                'Score': classification[0]['score']
            })


# Convert the detailed data to a DataFrame
detailed_df = pd.DataFrame(detailed_data)

# Save the detailed DataFrame to a CSV file
detailed_df.to_csv('results/ESG_BERT_highscores.csv', index=False)

# Print the detailed DataFrame
print(detailed_df)
