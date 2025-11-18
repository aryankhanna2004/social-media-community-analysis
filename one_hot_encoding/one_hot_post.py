import pandas as pd
import numpy as np

# Load the CSV of posts with their topic assignments
df = pd.read_csv("results/topic_modelling_output/posts_with_topics.csv")

# Assume each row is a post and has columns: 'subreddit', 'topic'
# We'll create a wide one-hot dataframe: columns = topic numbers
num_topics = df['topic'].nunique()
post_one_hot = pd.get_dummies(df['topic'], prefix='topic')

# Keep subreddit and post ID if needed
post_one_hot['subreddit'] = df['subreddit']
if 'post_id' in df.columns:
    post_one_hot['post_id'] = df['post_id']

# Rearrange columns: subreddit, post_id, then topics
cols = ['subreddit'] + (['post_id'] if 'post_id' in df.columns else []) + [c for c in post_one_hot.columns if c.startswith('topic_')]
post_one_hot = post_one_hot[cols]

# Save
post_one_hot.to_csv("results/one_hot_encoding/posts_one_hot.csv", index=False)
print(post_one_hot.head())
