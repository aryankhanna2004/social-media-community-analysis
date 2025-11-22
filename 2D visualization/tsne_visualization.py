import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_data():
    posts_df = pd.read_csv("results/topic_modelling_output/posts_with_topics.csv")
    with open("results/one_hot_encoding/heterogeneity_analysis.json", 'r') as f:
        info = json.load(f)
    return posts_df, info

def generate_map(posts_df, community_name, output_dir):
    print(f"Processing r/{community_name}...")
    
    community_posts = posts_df[posts_df['subreddit'] == community_name].copy()
    texts = community_posts['full_text'].fillna('').tolist()
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    
    n_components = min(50, len(community_posts))
    pca_result = PCA(n_components=n_components).fit_transform(embeddings)
    
    perplexity = min(30, len(community_posts) - 1)
    tsne_results = TSNE(n_components=2, verbose=0, perplexity=perplexity, random_state=42).fit_transform(pca_result)
    
    community_posts['x'] = tsne_results[:, 0]
    community_posts['y'] = tsne_results[:, 1]
    # Calculate topic counts and create labels
    topic_counts = community_posts['topic'].value_counts()
    
    def get_label(topic):
        count = topic_counts.get(topic, 0)
        if topic == -1:
            return f"Outliers (-1) (n={count})"
        return f"{topic} (n={count})"

    community_posts['topic_label'] = community_posts['topic'].apply(get_label)
    
    plt.figure(figsize=(14, 10))
    
    unique_topics = sorted(community_posts['topic'].unique())
    valid_topics = [t for t in unique_topics if t != -1]
    
    palette = {}
    hue_order = []
    if valid_topics:
        colors = sns.color_palette("tab20", len(valid_topics))
        for i, topic in enumerate(valid_topics):
            label = get_label(topic)
            palette[label] = colors[i]
            hue_order.append(label)
        
    outliers = community_posts[community_posts['topic'] == -1]
    if not outliers.empty:
        outlier_label = get_label(-1)
        plt.scatter(outliers['x'], outliers['y'], c='#d3d3d3', label=outlier_label, alpha=0.5, s=40)
        
    valid_posts = community_posts[community_posts['topic'] != -1]
    if not valid_posts.empty:
        sns.scatterplot(
            data=valid_posts, x='x', y='y', hue='topic_label', palette=palette, hue_order=hue_order,
            legend='full', alpha=0.8, s=70, edgecolor='white', linewidth=0.5
        )
    
    plt.title(f"2D Topic Map for r/{community_name}", fontsize=18, pad=20)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Topic ID")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"2d_map_{community_name}.png"), dpi=150)
    plt.close()

def main():
    output_dir = "results/visualization/2d_maps"
    os.makedirs(output_dir, exist_ok=True)
    
    posts_df, info = load_data()
    
    targets = [
        info['most_heterogeneous']['community'],
        info['most_homogeneous']['community']
    ]
    
    for community in targets:
        generate_map(posts_df, community, output_dir)

if __name__ == "__main__":
    main()
