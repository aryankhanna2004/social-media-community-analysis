import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import os

def load_one_hot_data():
    df = pd.read_csv("results/one_hot_encoding/one_hot_topics.csv", index_col=0)
    return df

def compute_cosine_similarity_matrix(df):
    vectors = df.values
    similarity_matrix = cosine_similarity(vectors)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=df.index,
        columns=df.index
    )
    return similarity_df

def get_community_pairs(similarity_df):
    pairs = []
    communities = similarity_df.index.tolist()
    
    for i, comm1 in enumerate(communities):
        for j, comm2 in enumerate(communities):
            if i < j:
                pairs.append({
                    'community_1': comm1,
                    'community_2': comm2,
                    'cosine_similarity': similarity_df.loc[comm1, comm2]
                })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('cosine_similarity', ascending=False)
    return pairs_df

def compute_fraction_shared_topics_matrix(df):
    communities = df.index.tolist()
    fraction_matrix = np.zeros((len(communities), len(communities)))
    
    for i, comm1 in enumerate(communities):
        for j, comm2 in enumerate(communities):
            if i == j:
                fraction_matrix[i, j] = 1.0
            else:
                vec1 = df.loc[comm1].values
                vec2 = df.loc[comm2].values
                
                shared_topics = np.sum((vec1 == 1) & (vec2 == 1))
                distinct_topics = np.sum((vec1 == 1) | (vec2 == 1))
                
                if distinct_topics > 0:
                    fraction_matrix[i, j] = shared_topics / distinct_topics
                else:
                    fraction_matrix[i, j] = 0.0
    
    fraction_df = pd.DataFrame(
        fraction_matrix,
        index=communities,
        columns=communities
    )
    return fraction_df

def analyze_community_heterogeneity(df):
    topic_counts = df.sum(axis=1)
    most_heterogeneous = topic_counts.idxmax()
    most_homogeneous = topic_counts.idxmin()
    
    return {
        'topic_counts': topic_counts.to_dict(),
        'most_heterogeneous': {
            'community': most_heterogeneous,
            'num_topics': int(topic_counts[most_heterogeneous])
        },
        'most_homogeneous': {
            'community': most_homogeneous,
            'num_topics': int(topic_counts[most_homogeneous])
        },
        'avg_topics_per_community': float(topic_counts.mean()),
        'std_topics_per_community': float(topic_counts.std())
    }

def apply_kmeans_clustering(df, k_values=None):
    if k_values is None:
        max_k = min(len(df.index) - 1, 10)
        k_values = list(range(2, max_k + 1))
    
    vectors = df.values
    results = {}
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)
        silhouette_avg = silhouette_score(vectors, cluster_labels)
        
        cluster_df = pd.DataFrame({
            'community': df.index,
            'cluster': cluster_labels
        })
        
        clusters = {}
        for cluster_id in range(k):
            communities_in_cluster = cluster_df[cluster_df['cluster'] == cluster_id]['community'].tolist()
            clusters[f'cluster_{cluster_id}'] = communities_in_cluster
        
        results[k] = {
            'silhouette_score': float(silhouette_avg),
            'inertia': float(kmeans.inertia_),
            'cluster_assignments': cluster_df.to_dict('records'),
            'clusters': clusters
        }
        
        print(f"\nK={k}:")
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Inertia: {kmeans.inertia_:.4f}")
        print(f"  Clusters:")
        for cluster_id, communities in clusters.items():
            print(f"    {cluster_id}: {communities}")
    
    return results

def main():
    print("=" * 60)
    print("Community Analysis: Cosine Similarity & K-means Clustering")
    print("=" * 60)
    
    print("\n1. Loading one-hot encoded data...")
    df = load_one_hot_data()
    print(f"   Loaded {len(df)} communities with {len(df.columns)} topics")
    
    print("\n2. Computing cosine similarity for all community pairs...")
    similarity_matrix = compute_cosine_similarity_matrix(df)
    pairs_df = get_community_pairs(similarity_matrix)
    
    print(f"\n   Top 5 most similar community pairs:")
    print(pairs_df.head().to_string(index=False))
    
    print(f"\n   Bottom 5 least similar community pairs:")
    print(pairs_df.tail().to_string(index=False))
    
    print("\n3. Analyzing community heterogeneity...")
    heterogeneity = analyze_community_heterogeneity(df)
    print(f"\n   Most heterogeneous: {heterogeneity['most_heterogeneous']['community']} ({heterogeneity['most_heterogeneous']['num_topics']} topics)")
    print(f"   Most homogeneous: {heterogeneity['most_homogeneous']['community']} ({heterogeneity['most_homogeneous']['num_topics']} topics)")
    print(f"   Average topics per community: {heterogeneity['avg_topics_per_community']:.2f}")
    print(f"   Std deviation: {heterogeneity['std_topics_per_community']:.2f}")
    print(f"\n   Topics per community:")
    for comm, count in sorted(heterogeneity['topic_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"     {comm}: {count} topics")
    
    print("\n4. Computing fraction of shared topics matrix...")
    fraction_matrix = compute_fraction_shared_topics_matrix(df)
    print(f"\n   Fraction of shared topics matrix preview:")
    print(fraction_matrix.head().to_string())
    
    print("\n5. Applying K-means clustering with different K values...")
    clustering_results = apply_kmeans_clustering(df)
    
    best_k = max(clustering_results.items(), key=lambda x: x[1]['silhouette_score'])
    
    print(f"Most heterogeneous: {heterogeneity['most_heterogeneous']['community']} ({heterogeneity['most_heterogeneous']['num_topics']} topics)")
    print(f"Most homogeneous: {heterogeneity['most_homogeneous']['community']} ({heterogeneity['most_homogeneous']['num_topics']} topics)")
    print(f"Best K: {best_k[0]} (silhouette: {best_k[1]['silhouette_score']:.4f})")
    
    os.makedirs("results/one_hot_encoding", exist_ok=True)
    
    similarity_matrix.to_csv("results/one_hot_encoding/cosine_similarity_matrix.csv")
    pairs_df.to_csv("results/one_hot_encoding/community_pairs_similarity.csv", index=False)
    fraction_matrix.to_csv("results/one_hot_encoding/fraction_shared_topics_matrix.csv")
    
    with open("results/one_hot_encoding/kmeans_clustering_results.json", "w") as f:
        json.dump(clustering_results, f, indent=2)
    
    clustering_summary = pd.DataFrame([
        {
            'k': k,
            'silhouette_score': results['silhouette_score'],
            'inertia': results['inertia'],
            'num_clusters': k
        }
        for k, results in clustering_results.items()
    ])
    clustering_summary.to_csv("results/one_hot_encoding/kmeans_clustering_summary.csv", index=False)
    
    with open("results/one_hot_encoding/heterogeneity_analysis.json", "w") as f:
        json.dump(heterogeneity, f, indent=2)

if __name__ == "__main__":
    main()
