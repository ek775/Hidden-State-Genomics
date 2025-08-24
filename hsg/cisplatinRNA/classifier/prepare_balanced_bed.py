import argparse
import pandas as pd
import random
from pathlib import Path

def read_and_filter_bed(filepath, target_length=100):
    """Read a BED file and return only entries with the desired length."""
    df = pd.read_csv(filepath, sep='\t', header=None, comment='#')
    df.columns = [f'col{i}' for i in range(df.shape[1])]
    df = df[df['col2'] - df['col1'] == target_length]
    return df

def label_and_sample(df, label, n=None, seed=42):
    """Add a label and optionally sample the DataFrame."""
    df = df.copy()
    df['label'] = label
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df

def main(pos_bed, neg_bed, output_file, neg_ratio=1.0, length=100):
    print(f"Reading positive examples from {pos_bed}")
    pos_df = read_and_filter_bed(pos_bed, target_length=length)
    num_pos = len(pos_df)
    print(f"Found {num_pos} positive sequences of {length}bp")

    print(f"Reading negative examples from {neg_bed}")
    neg_df = read_and_filter_bed(neg_bed, target_length=length)
    num_neg = int(num_pos * neg_ratio)
    print(f"Sampling {num_neg} negative sequences (ratio {neg_ratio}) from {len(neg_df)} total")

    pos_df = label_and_sample(pos_df, label=1)
    neg_df = label_and_sample(neg_df, label=0, n=num_neg)

    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=42)  # shuffle

    print(f"Saving combined dataset ({len(combined_df)} sequences) to {output_file}")
    combined_df.to_csv(output_file, sep='\t', header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample negative BED file and merge with positives for MLP training.")
    parser.add_argument("positive_bed", help="Path to BED file of positive sequences")
    parser.add_argument("negative_bed", help="Path to BED file of negative sequences")
    parser.add_argument("output", help="Output path for merged and labeled sequences")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="Ratio of negatives to positives (default 1.0)")
    parser.add_argument("--length", type=int, default=100, help="Target sequence length (default 100bp)")
    args = parser.parse_args()

    main(args.positive_bed, args.negative_bed, args.output, args.neg_ratio, args.length)
