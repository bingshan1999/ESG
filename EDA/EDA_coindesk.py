import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# Load your data
df = pd.read_csv('../data/coindesk_btc.csv')

# Check data types and missing values
print(df.info())

# Summary statistics
#print(df.describe())
#print(df.head())

# Data Cleaning: Check for duplicates
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Data Cleaning: Check for missing values and handle them
print("\nMissing Values:")
print(df['content'].isnull().sum())

# Identify rows with missing values in the specified columns
rows_with_missing_values = df[df['content'].isnull()]
print("\nRows with Missing Values in 'content' Column:")
print(rows_with_missing_values)

df = df.dropna(subset=['content'])

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