# dbscan_cluster_analysis.py (Amino Acid & BLOSUM Version with Progress Bars)

import pandas as pd
import numpy as np
import argparse
import os
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.Align import substitution_matrices
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Translate DNA to amino acid sequences ---
def translate_sequences(dna_seqs):
    aa_seqs = []
    for dna in tqdm(dna_seqs, desc="Translating sequences"):
        try:
            seq = Seq(dna)
            translated = str(seq.translate(to_stop=True))
            aa_seqs.append(translated)
        except Exception as e:
            aa_seqs.append("")  # fallback for problematic seqs
    return aa_seqs

# --- Compute pairwise BLOSUM distances ---
def compute_blosum_distance_matrix(sequences):
    matrix = substitution_matrices.load("BLOSUM62")
    n = len(sequences)
    dist_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Computing distance matrix"):
        for j in range(i+1, n):
            s1, s2 = sequences[i], sequences[j]
            aln = pairwise2.align.globalds(s1, s2, matrix, -10, -0.5, one_alignment_only=True)
            score = aln[0].score if aln else 0
            dist = -score  # more similar → less distance
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix

# --- t-SNE visualization ---
def plot_tsne(dist_matrix, labels, output_path):
    tsne = TSNE(n_components=2, metric="precomputed", random_state=42)
    reduced = tsne.fit_transform(dist_matrix)
    df = pd.DataFrame({"Dim1": reduced[:, 0], "Dim2": reduced[:, 1], "Category": labels})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Category", alpha=0.7)
    plt.title("t-SNE of Amino Acid Sequences (BLOSUM Distances)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to classifier predictions CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to store results")
    parser.add_argument("--eps", type=float, default=50.0, help="DBSCAN epsilon (distance threshold)")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN minimum samples")
    parser.add_argument(
        "--categories", type=str, default="True Positive,True Negative,False Positive,False Negative",
        help="Comma-separated list of categories to analyze (e.g. 'True Positive,False Positive')"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if "category" not in df.columns:
        raise ValueError("Input CSV must contain a 'category' column. Please ensure it comes from the MLP classifier.")

    # Parse categories from input argument
    categories = [cat.strip() for cat in args.categories.split(",")]

    print(f"Analyzing categories: {categories}")

    all_aa_seqs = []
    all_labels = []
    all_indices = []

    for cat in categories:
        sub_df = df[df["category"] == cat].copy()
        if sub_df.empty:
            print(f"No sequences found for category '{cat}', skipping.")
            continue
        
        aa_seqs = translate_sequences(sub_df["sequence"].tolist())
        sub_df["aa_seq"] = aa_seqs
        sub_df = sub_df[sub_df["aa_seq"].str.len() > 0]  # Remove empty translations

        if len(sub_df) == 0:
            print(f"No translatable sequences for category {cat}, skipping.")
            continue

        print(f"Computing distances for {cat} ({len(sub_df)} sequences)...")
        dist_matrix = compute_blosum_distance_matrix(sub_df["aa_seq"].tolist())

        print(f"Running DBSCAN for {cat}...")
        db = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="precomputed")
        cluster_labels = db.fit_predict(dist_matrix)
        sub_df["cluster_label"] = cluster_labels

        clustered_df = sub_df[sub_df["cluster_label"] != -1].copy()
        if len(clustered_df) == 0:
            print(f"No clusters found in {cat} after DBSCAN.")
            continue

        # Save clustered results
        out_path = os.path.join(args.output_dir, f"{cat.replace(' ', '_')}_clusters_blosum.tsv")
        clustered_df.to_csv(out_path, sep="\t", index=False)

        all_aa_seqs.extend(clustered_df["aa_seq"].tolist())
        all_labels.extend([cat] * len(clustered_df))
        all_indices.extend(clustered_df.index.tolist())

    if len(all_aa_seqs) > 1:
        print("Generating t-SNE visualization...")
        tsne_dist_matrix = compute_blosum_distance_matrix(all_aa_seqs)
        tsne_out_path = os.path.join(args.output_dir, "tsne_blosum_plot.png")
        plot_tsne(tsne_dist_matrix, all_labels, tsne_out_path)
        print(f"Saved t-SNE plot to {tsne_out_path}")
    else:
        print("Not enough clustered sequences to generate t-SNE.")

if __name__ == "__main__":
    main()
