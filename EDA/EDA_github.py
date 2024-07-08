import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

# Load your data
df = pd.read_csv('../data/github_btc.csv')

# Check data types and missing values
print(df.info())

# Data Cleaning: Check for duplicates
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Data Cleaning: Check for missing values and handle them
print("\nMissing Values:")
print(df['issue_body_and_comments'].isnull().sum())

# Identify rows with missing values in the specified columns
# rows_with_missing_values = df[df['issue_body_and_comments'].isnull()]
# print("\nRows with Missing Values in 'content' Column:")
# print(rows_with_missing_values)

df = df.dropna(subset=['issue_body_and_comments'])

# Text Analysis: Distribution of issue lengths
df['word count'] = df['issue_body_and_comments'].apply(lambda x: len(x.split()))
#print("\nDistribution of Issue Lengths:")
#print(df['issue_body_and_comments'].describe())

# Plotting the distribution of issue lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['word count'], bins=30, kde=True)
plt.title('Distribution of Issue Lengths')
plt.xlabel('Issue Length')
plt.ylabel('Frequency')
#plt.show()

# Define stop words
stop_words = set(stopwords.words('english'))

# Function to remove stop words
def remove_stop_words(words):
    return [word for word in words if word not in stop_words]

# Most common words in titles
all_titles = ' '.join(df['issue_title']).lower().split()
filtered_titles = remove_stop_words(all_titles)
title_word_counts = Counter(filtered_titles)

print("\nMost Common Words (non-stopwords) in Issue Titles:")
print(title_word_counts.most_common(5))

# Temporal Analysis: Issues per year
df['created_at'] = pd.to_datetime(df['created_at'])
df['year'] = df['created_at'].dt.year
issues_per_year = df['year'].value_counts().sort_index()
# print("\Issues Published Per Year:")
# print(issues_per_year)

# Plotting the number of articles published per year
plt.figure(figsize=(10, 6))
issues_per_year.plot(kind='bar')
plt.title('Number of Issues Published Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Issues')
#plt.show()

# Count the number of rows with 0 comments
num_zero_comments = df[df['number_of_comments'] == 0].shape[0]
print(f"Number of rows with 0 comments: {num_zero_comments}")

plt.figure(figsize=(10, 6))
df['number_of_comments'].plot(kind='hist', bins=1000)
plt.title('Distribution of Number of Comments')
plt.xlabel('Number of Comments')
plt.ylabel('Frequency')
#plt.show()


text = ' '.join(df['issue_body_and_comments'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

#Plot Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Issue Contents')
#plt.show()

#Perform Sentiment Analysis
df['polarity'] = df['issue_body_and_comments'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['issue_body_and_comments'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Summary of Sentiment Analysis
# print("\nSummary of Sentiment Analysis:")
# print(df[['polarity', 'subjectivity']].describe())

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