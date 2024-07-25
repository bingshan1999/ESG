import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import re

def clean_article(text):
    # Define keywords to identify unwanted sections
    keywords = [
        "CoinDesk", "privacy policy", "terms of use", "do not sell my personal information",
        "award-winning media outlet", "Bullish group", "editorial policies", "regulatory compliance"
    ]
    
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Function to check if any keyword is in the sentence
    def contains_keyword(sentence):
        return any(keyword.lower() in sentence.lower() for keyword in keywords)

    # Iterate from the end and find the position to start keeping sentences
    keep_index = len(sentences)
    lookahead = 2  # Number of sentences to look ahead
    
    for i in range(len(sentences) - 1, -1, -1):
        if contains_keyword(sentences[i]):
            # Check lookahead sentences
            lookahead_detected = False
            for j in range(1, lookahead + 1):
                if i - j >= 0 and contains_keyword(sentences[i - j]):
                    lookahead_detected = True
                    break
            
            if not lookahead_detected:
                keep_index = i
                break
    
    # Keep sentences from the start to the identified position
    cleaned_text = ' '.join(sentences[:keep_index])
    return cleaned_text


# Load your data using pandas
file_path = '../data/coindesk_btc.csv'
df = pd.read_csv(file_path)

# Drop rows with missing content
df = df.dropna(subset=['content'])

# Remove duplicates
df = df.drop_duplicates(subset=['content'])

# Apply the clean_article function to the 'content' column
df['content'] = df['content'].apply(clean_article)

# Save the cleaned DataFrame to a new CSV file
df.to_csv("../data/cleaned_coindesk_btc.csv", index=False)


# Text Analysis: Distribution of article lengths
df['word count'] = df['content'].apply(lambda x: len(x.split()))
print("\nDistribution of Article Lengths:")
print(df['word count'].describe())

# Plotting the distribution of article lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['word count'], bins=30, kde=True)
plt.title('Distribution of Article Lengths')
plt.xlabel('Content Length')
plt.ylabel('Frequency')
#plt.show()

# Most common words in titles
all_titles = ' '.join(df['title']).lower().split()
title_word_counts = Counter(all_titles)
print("\nMost Common Words in Titles:")
print(title_word_counts.most_common(5))

# Temporal Analysis: Articles per year
articles_per_year = df['year'].value_counts().sort_index()
print("\nArticles Published Per Year:")
print(articles_per_year)

# Plotting the number of articles published per year
plt.figure(figsize=(10, 6))
articles_per_year.plot(kind='bar')
plt.title('Number of Articles Published Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Articles')
#plt.show()

text = ' '.join(df['content'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Article Contents')
#plt.show()

# Perform Sentiment Analysis
df['polarity'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['content'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Summary of Sentiment Analysis
print("\nSummary of Sentiment Analysis:")
print(df[['polarity', 'subjectivity']].describe())

# Plotting Sentiment Analysis
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Polarity distribution
sns.histplot(df['polarity'], bins=30, kde=True, ax=ax[0])
ax[0].set_title('Distribution of Polarity')
ax[0].set_xlabel('Polarity')
ax[0].set_ylabel('Frequency')

# Subjectivity distribution
sns.histplot(df['subjectivity'], bins=30, kde=True, ax=ax[1])
ax[1].set_title('Distribution of Subjectivity')
ax[1].set_xlabel('Subjectivity')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()