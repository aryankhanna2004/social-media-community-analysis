import pandas as pd

# Load your file
df = pd.read_csv("results/topic_modelling_output/posts_with_topics.csv", sep=",")

# Group by subreddit (community) and topic number
topic_counts = df.groupby(["subreddit", "topic"]).size().unstack(fill_value=0)

# Apply a threshold: e.g., only keep topics with >= 5 posts in that subreddit
threshold = 5
one_hot = (topic_counts >= threshold).astype(int)

# View the first few rows
print(one_hot.head())

# Save the one-hot encoded DataFrame to a CSV file
one_hot.to_csv("results/one_hot_encoding/one_hot_topics.csv", index=True)

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Suppose 'one_hot' is your DataFrame
# Compute cosine similarity between communities
similarity_matrix = cosine_similarity(one_hot)

# Convert to a DataFrame for easier handling
similarity_df = pd.DataFrame(
    similarity_matrix, 
    index=one_hot.index, 
    columns=one_hot.index
)

# View first few rows
print(similarity_df.head())
