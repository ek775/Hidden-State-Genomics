# mlp_bed_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pyfaidx import Fasta
from Bio.Seq import Seq
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import datetime
import argparse
import os
from tqdm import tqdm
import random  # For shuffling labels

# ---------- Config ----------
SEQUENCE_LENGTH = 100
INPUT_DIM = SEQUENCE_LENGTH * 4
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3


def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return torch.tensor([mapping.get(base, [0, 0, 0, 0]) for base in seq], dtype=torch.float32)


def fetch_sequence(chrom, start, end, strand, fasta):
    seq = fasta[chrom][start:end].seq.upper()
    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
    return seq


def process_labeled_bed_file(bed_path, fasta):
    data = []
    fasta_chroms = set(fasta.keys())  # Get set of all chromosomes in FASTA
    with open(bed_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            strand = parts[5] if len(parts) > 5 else '+'
            label = int(parts[-1])
            
            if chrom not in fasta_chroms:
                print(f"Warning: Chromosome {chrom} not found in FASTA. Skipping this entry.")
                continue
            
            seq = fetch_sequence(chrom, start, end, strand, fasta)
            encoded = one_hot_encode(seq).flatten()
            data.append((encoded, label))
    return data


def pad_or_truncate(tensor, target_len):
    current_len = tensor.shape[0]
    if current_len > target_len:
        return tensor[:target_len]
    elif current_len < target_len:
        pad_len = target_len - current_len
        return F.pad(tensor, (0, pad_len))
    return tensor


def shuffle_labels(data):
    sequences = [x for x, y in data]
    labels = [y for x, y in data]
    random.shuffle(labels)
    return list(zip(sequences, labels))


class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, torch.tensor(y, dtype=torch.float32)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)).squeeze(1)
        return x


def log_metrics(log_path, roc, cm, report):
    with open(log_path, 'a') as log:
        log.write(f"Run at {datetime.datetime.now()}\n")
        log.write(f"ROC AUC Score: {roc:.4f}\n")
        log.write("Confusion Matrix:\n")
        log.write("                 Predicted Negative    Predicted Positive\n")
        log.write(f"Actual Negative     {cm[0][0]:>8} (TN)         {cm[0][1]:>8} (FP)\n")
        log.write(f"Actual Positive     {cm[1][0]:>8} (FN)         {cm[1][1]:>8} (TP)\n")
        log.write("Classification Report:\n")
        log.write(report)
        log.write("\n" + "="*60 + "\n")


def save_predictions(output_path, y_true, y_scores):
    df = pd.DataFrame({'true_label': y_true, 'score': y_scores})
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train MLP classifier on one-hot encoded genomic sequences")
    parser.add_argument("--log_file", type=str, default="mlp_classification_report.log", help="Output log file for performance metrics")
    parser.add_argument("--labeled_bed", type=str, required=True, help="Path to labeled BED file")
    parser.add_argument("--fasta", type=str, required=True, help="Path to reference FASTA file")
    parser.add_argument("--pred_output", type=str, default=None, help="Optional CSV file to save raw predictions")
    parser.add_argument("--shuffle_labels", action='store_true', help="Train model with shuffled labels for sanity check")
    args = parser.parse_args()

    fasta = Fasta(args.fasta)

    print("Processing labeled BED file...")
    labeled_data = process_labeled_bed_file(args.labeled_bed, fasta)
    all_data = [(pad_or_truncate(x, INPUT_DIM), y) for x, y in labeled_data]

    if args.shuffle_labels:
        print("Shuffling labels for sanity check...")
        all_data = shuffle_labels(all_data)

    # Split dataset, stratify on labels
    train_data, test_data = train_test_split(all_data, test_size=0.3, stratify=[y for _, y in all_data])
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    train_loader = DataLoader(SequenceDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SequenceDataset(test_data), batch_size=BATCH_SIZE)

    model = MLPClassifier(INPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating", leave=False):
            output = model(X)
            y_true.extend(y.numpy())
            y_scores.extend(output.numpy())

    y_pred = [1 if s > 0.5 else 0 for s in y_scores]

    # Metrics
    roc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print(f"\nROC AUC Score: {roc:.4f}")
    print("Confusion Matrix:")
    print("                 Predicted Negative    Predicted Positive")
    print(f"Actual Negative     {cm[0][0]:>8} (TN)         {cm[0][1]:>8} (FP)")
    print(f"Actual Positive     {cm[1][0]:>8} (FN)         {cm[1][1]:>8} (TP)")
    print("Classification Report:")
    print(report)

    # Log to file
    log_metrics(args.log_file, roc, cm, report)

    # Optionally save predictions
    if args.pred_output:
        save_predictions(args.pred_output, y_true, y_scores)


if __name__ == "__main__":
    main()