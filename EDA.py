import pandas as pd

# Load your data
df = pd.read_csv('reddit_posts.csv')

# Check data types and missing values
print(df.info())

# Summary statistics
print(df.describe())
print(df.head())