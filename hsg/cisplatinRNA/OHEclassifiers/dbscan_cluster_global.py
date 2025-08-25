# dbscan_cluster_global.py


import pandas as pd
import numpy as np
import argparse
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- One-hot encoding for DNA sequences ---
def one_hot_encode_sequences(sequences):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    encoded = [np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq]).flatten() for seq in sequences]
    return np.stack(encoded)

# --- DBSCAN clustering for all sequences ---
def cluster_all_sequences(df, eps=5, min_samples=3):
    sequences = df["sequence"].tolist()
    encoded = one_hot_encode_sequences(sequences)
    
    print(f"Clustering {len(sequences)} sequences globally...")
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(encoded)
    labels = db.labels_
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered["cluster_label"] = labels
    
    # Separate clustered vs noise points
    clustered_df = df_clustered[df_clustered["cluster_label"] != -1].copy()
    noise_df = df_clustered[df_clustered["cluster_label"] == -1].copy()
    
    print(f"Found {len(np.unique(labels[labels != -1]))} clusters")
    print(f"Clustered sequences: {len(clustered_df)}")
    print(f"Noise sequences: {len(noise_df)}")
    
    return clustered_df, noise_df, encoded[labels != -1], labels[labels != -1]

# --- Cluster composition analysis ---
def analyze_cluster_composition(clustered_df):
    """Analyze the category composition of each cluster"""
    composition_stats = []
    
    for cluster_id in sorted(clustered_df["cluster_label"].unique()):
        cluster_data = clustered_df[clustered_df["cluster_label"] == cluster_id]
        total_seqs = len(cluster_data)
        
        # Count sequences by category
        category_counts = cluster_data["category"].value_counts()
        
        composition = {
            "cluster_id": cluster_id,
            "total_sequences": total_seqs,
            "dominant_category": category_counts.index[0],
            "dominant_count": category_counts.iloc[0],
            "dominant_percentage": (category_counts.iloc[0] / total_seqs) * 100
        }
        
        # Add counts for each category
        for category in ["True Positive", "True Negative", "False Positive", "False Negative"]:
            composition[f"{category.replace(' ', '_')}_count"] = category_counts.get(category, 0)
            composition[f"{category.replace(' ', '_')}_percentage"] = (category_counts.get(category, 0) / total_seqs) * 100
        
        composition_stats.append(composition)
    
    return pd.DataFrame(composition_stats)

# --- Compute cluster centroids and inter-cluster distances ---
def compute_cluster_centroids_and_distances(clustered_df, encoded_sequences, cluster_labels):
    """Compute centroids for each cluster and distances between them"""
    unique_clusters = np.unique(cluster_labels)
    centroids = {}
    
    # Compute centroids
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        centroids[cluster_id] = np.mean(encoded_sequences[mask], axis=0)
    
    # Compute pairwise distances
    distance_rows = []
    for i, c1 in enumerate(unique_clusters):
        for j, c2 in enumerate(unique_clusters):
            if i >= j:
                continue
            cos_dist = cosine_distances([centroids[c1]], [centroids[c2]])[0][0]
            euc_dist = euclidean_distances([centroids[c1]], [centroids[c2]])[0][0]
            
            distance_rows.append({
                "Cluster1": c1,
                "Cluster2": c2,
                "CosineDistance": cos_dist,
                "EuclideanDistance": euc_dist
            })
    
    return pd.DataFrame(distance_rows)

# --- t-SNE visualization colored by cluster and category ---
def plot_tsne_visualizations(clustered_df, encoded_sequences, output_dir):
    """Create t-SNE plots colored by cluster and by original category"""
    
    # Compute t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(encoded_sequences)-1))
    reduced = tsne.fit_transform(encoded_sequences)
    
    # Create dataframe for plotting
    plot_df = clustered_df.copy()
    plot_df["tsne_x"] = reduced[:, 0]
    plot_df["tsne_y"] = reduced[:, 1]
    
    # Plot 1: Colored by cluster
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(plot_df["tsne_x"], plot_df["tsne_y"], 
                         c=plot_df["cluster_label"], cmap='tab20', alpha=0.7, s=50)
    plt.colorbar(scatter)
    plt.title("t-SNE: Colored by Cluster ID")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # Plot 2: Colored by original category
    plt.subplot(1, 2, 2)
    categories = plot_df["category"].unique()
    colors = ['red', 'blue', 'green', 'orange']
    for i, category in enumerate(categories):
        mask = plot_df["category"] == category
        plt.scatter(plot_df[mask]["tsne_x"], plot_df[mask]["tsne_y"], 
                   c=colors[i % len(colors)], label=category, alpha=0.7, s=50)
    plt.legend()
    plt.title("t-SNE: Colored by Original Category")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_global_clusters.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- Create cluster summary heatmap ---
def plot_cluster_category_heatmap(clustered_df, output_dir):
    """Create heatmap showing category distribution across clusters"""
    
    # Create pivot table
    pivot_table = clustered_df.groupby(['cluster_label', 'category']).size().unstack(fill_value=0)
    
    # Convert to percentages within each cluster
    pivot_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(10, max(6, len(pivot_percentage) * 0.4)))
    sns.heatmap(pivot_percentage, annot=True, fmt='.1f', cmap='Blues', 
                cbar_kws={'label': 'Percentage of Sequences in Cluster'})
    plt.title('Category Distribution Across Clusters (%)')
    plt.xlabel('Original Category')
    plt.ylabel('Cluster ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_category_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Global DBSCAN clustering of DNA sequences across all categories")
    parser.add_argument("--input_csv", required=True, help="Path to classifier predictions CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to store results")
    parser.add_argument("--eps", type=float, default=5, help="DBSCAN epsilon parameter")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min samples parameter")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    if "category" not in df.columns:
        raise ValueError("Input CSV must contain a 'category' column. Please ensure you are using the correct MLP classifier output file.")
    
    if "sequence" not in df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column.")
    
    print(f"Loaded {len(df)} sequences across {df['category'].nunique()} categories")
    print("Category distribution:")
    print(df["category"].value_counts())
    
    # Perform global clustering
    clustered_df, noise_df, encoded_sequences, cluster_labels = cluster_all_sequences(
        df, eps=args.eps, min_samples=args.min_samples
    )
    
    if len(clustered_df) == 0:
        print("No clusters found! Try adjusting eps and min_samples parameters.")
        return
    
    # Save clustered sequences
    clustered_output = os.path.join(args.output_dir, "global_clusters.tsv")
    clustered_df.to_csv(clustered_output, sep="\t", index=False)
    print(f"Saved clustered sequences to {clustered_output}")
    
    # Save noise sequences
    if len(noise_df) > 0:
        noise_output = os.path.join(args.output_dir, "noise_sequences.tsv")
        noise_df.to_csv(noise_output, sep="\t", index=False)
        print(f"Saved noise sequences to {noise_output}")
    
    # Analyze cluster composition
    composition_df = analyze_cluster_composition(clustered_df)
    composition_output = os.path.join(args.output_dir, "cluster_composition.tsv")
    composition_df.to_csv(composition_output, sep="\t", index=False)
    print(f"Saved cluster composition analysis to {composition_output}")
    
    # Compute inter-cluster distances
    distances_df = compute_cluster_centroids_and_distances(clustered_df, encoded_sequences, cluster_labels)
    distances_output = os.path.join(args.output_dir, "inter_cluster_distances.tsv")
    distances_df.to_csv(distances_output, sep="\t", index=False)
    print(f"Saved inter-cluster distances to {distances_output}")
    
    # Create visualizations
    print("Creating visualizations...")
    plot_tsne_visualizations(clustered_df, encoded_sequences, args.output_dir)
    print(f"Saved t-SNE plots to {args.output_dir}/tsne_global_clusters.png")
    
    plot_cluster_category_heatmap(clustered_df, args.output_dir)
    print(f"Saved cluster composition heatmap to {args.output_dir}/cluster_category_heatmap.png")
    
    # Print summary statistics
    print("\n=== CLUSTERING SUMMARY ===")
    print(f"Total sequences: {len(df)}")
    print(f"Clustered sequences: {len(clustered_df)}")
    print(f"Noise sequences: {len(noise_df)}")
    print(f"Number of clusters: {clustered_df['cluster_label'].nunique()}")
    print(f"Clustering efficiency: {len(clustered_df)/len(df)*100:.1f}%")
    
    #print("\n=== CLUSTER SIZES ===")
    cluster_sizes = clustered_df["cluster_label"].value_counts().sort_index()
    #for cluster_id, size in cluster_sizes.items():
        #print(f"Cluster {cluster_id}: {size} sequences")
    
    #print("\n=== CLUSTER PURITY ===")
    #for _, row in composition_df.iterrows():
        #print(f"Cluster {int(row['cluster_id'])}: {row['dominant_percentage']:.1f}% {row['dominant_category']} "
              #f"({int(row['dominant_count'])}/{int(row['total_sequences'])} sequences)")

if __name__ == "__main__":
    main()