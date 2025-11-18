import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text  # <-- new import

# Load the CSV file directly
df = pd.read_csv("results\one_hot_encoding\posts_one_hot.csv")   # <-- put the correct filename

print(df.head())
print("Number of posts:", len(df))

import pandas as pd

df = pd.read_csv("results\one_hot_encoding\posts_one_hot.csv")

# Select only topic columns (boolean → ints)
topic_cols = [c for c in df.columns if c.startswith("topic_")]
df[topic_cols] = df[topic_cols].astype(int)

# Rename subreddit to community (optional, for consistency)
df = df.rename(columns={"subreddit": "community"})

print(df.columns)


from sklearn.cluster import KMeans

#save graphs
def plot_tsne(df_exp, community_name, n_clusters=6, top_n_topics=3):
    df_comm = df_exp[df_exp['community'] == community_name].reset_index(drop=True)

    if len(df_comm) < 10:
        print(f"Not enough posts for {community_name} (n={len(df_comm)})")
        return None

    X = df_comm.drop(columns=['community', 'post_id']).values
    topic_cols = [c for c in df_comm.columns if c.startswith("topic_")]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_2d)
    centers = kmeans.cluster_centers_

    df_comm['cluster'] = labels

    # Compute top topics per cluster
    cluster_topics = df_comm.groupby('cluster')[topic_cols].mean()
    cluster_top_topics = {}
    for cluster_id, row in cluster_topics.iterrows():
        top_topics = row.sort_values(ascending=False).head(top_n_topics).index.tolist()
        cluster_top_topics[cluster_id] = top_topics

    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                          c=labels, cmap='tab10', s=50)

    # Annotate cluster centers with top topics
    texts = []
    for i, (cx, cy) in enumerate(centers):
        topics_str = ", ".join(cluster_top_topics[i])
        texts.append(plt.text(cx, cy, f"{i}\n{topics_str}",
                              fontsize=12, fontweight='bold',
                              ha='center', va='center',
                              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round')))

    # Adjust text to avoid overlaps
    adjust_text(texts, force_points=0.2, force_text=0.2)

    plt.title(f"t-SNE + KMeans Clusters for {community_name}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.show()

    return df_comm, labels

df_comm, labels = plot_tsne(df, "jpop")
df_comm, labels = plot_tsne(df, "classicalmusic")
