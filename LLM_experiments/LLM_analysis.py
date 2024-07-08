import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# # Load your data
# df = pd.read_csv('data/coindesk_btc_sentences_with_esg.csv')

# #pd.set_option('display.max_colwidth', None)
# # Check data types and missing values
# print(df.describe())


from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)