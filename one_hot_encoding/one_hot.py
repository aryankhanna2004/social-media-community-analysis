import pandas as pd
import numpy as np

# Load data and group by community and topic
df = pd.read_csv("results/topic_modelling_output/posts_with_topics.csv", sep=",")
topic_counts = df.groupby(["subreddit", "topic"]).size().unstack(fill_value=0)
community_totals = df.groupby("subreddit").size()

# Apply percentage threshold (e.g., 3 = 3%)
threshold_percentage = 3
min_posts = (community_totals * threshold_percentage / 100).apply(np.ceil).astype(int)

# Create one-hot encoding: 1 if topic count >= threshold, else 0
one_hot = topic_counts.copy()
for community in topic_counts.index:
    one_hot.loc[community] = (topic_counts.loc[community] >= min_posts[community]).astype(int)

print(f"Threshold: {threshold_percentage}% per community")
print(f"Average topics per community: {one_hot.sum(axis=1).mean():.2f}")
print("\nOne-hot encoding preview:")
print(one_hot.head())

one_hot.to_csv("results/one_hot_encoding/one_hot_topics.csv", index=True)
