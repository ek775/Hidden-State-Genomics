# dbscan_cluster_analysis.py

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
    mapping = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}
    encoded = [np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq]).flatten() for seq in sequences]
    return np.stack(encoded)

# --- DBSCAN clustering ---
def cluster_category(sequences, eps=40, min_samples=3):
    encoded = one_hot_encode_sequences(sequences)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(encoded)
    labels = db.labels_
    filtered = encoded[labels != -1]
    label_ids = labels[labels != -1]
    return filtered, label_ids, db

# --- Centroid distance analysis ---
def compute_centroids_and_distances(embeddings_dict):
    centroids = {cat: np.mean(vecs, axis=0) for cat, vecs in embeddings_dict.items()}
    cats = list(centroids.keys())
    rows = []
    for i in range(len(cats)):
        for j in range(len(cats)):
            if i >= j:
                continue
            c1, c2 = cats[i], cats[j]
            cos = cosine_distances([centroids[c1]], [centroids[c2]])[0][0]
            euc = euclidean_distances([centroids[c1]], [centroids[c2]])[0][0]
            rows.append({"Category1": c1, "Category2": c2, "CosineDistance": cos, "EuclideanDistance": euc})
    return pd.DataFrame(rows)

# --- t-SNE visualization ---
def plot_tsne(embeddings_dict, output_file):
    all_vecs = []
    labels = []
    for category, vecs in embeddings_dict.items():
        all_vecs.extend(vecs)
        labels.extend([category]*len(vecs))
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(np.array(all_vecs))
    df = pd.DataFrame({"Dim1": reduced[:, 0], "Dim2": reduced[:, 1], "Category": labels})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Category", alpha=0.7)
    plt.title("t-SNE of Clustered Sequences (Noise Filtered)")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to classifier predictions CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to store results")
    parser.add_argument("--eps", type=float, default=5, help="DBSCAN epsilon")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if "category" not in df.columns:
        raise ValueError("Input CSV must contain a 'category' column. Please ensure you are using the correct MLP classifier output file.")

    categories = ["True Positive", "True Negative", "False Positive", "False Negative"]
    embeddings_dict = {}

    for cat in categories:
        sub_df = df[df["category"] == cat]
        print(f"Clustering category: {cat} with {len(sub_df)} sequences")
        encoded, labels, db = cluster_category(sub_df["sequence"].tolist(), eps=args.eps, min_samples=args.min_samples)
        if len(encoded) == 0:
            print(f"No clustered points for {cat}, skipping.")
            continue
        embeddings_dict[cat] = encoded
        out_path = os.path.join(args.output_dir, f"{cat.replace(' ', '_')}_clusters.tsv")
        cluster_df = sub_df.iloc[db.labels_ != -1].copy()
        cluster_df["cluster_label"] = labels
        cluster_df.to_csv(out_path, sep="\t", index=False)

    distance_df = compute_centroids_and_distances(embeddings_dict)
    distance_df.to_csv(os.path.join(args.output_dir, "centroid_distances.tsv"), sep="\t", index=False)

    tsne_path = os.path.join(args.output_dir, "tsne_plot.png")
    plot_tsne(embeddings_dict, tsne_path)
    print(f"Saved t-SNE to {tsne_path}")

if __name__ == "__main__":
    main()

