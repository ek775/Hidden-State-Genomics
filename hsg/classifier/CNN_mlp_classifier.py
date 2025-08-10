# CNN_mlp_classifier.py

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

# ---------- Classes ----------
class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x[0], torch.tensor(y, dtype=torch.float32)

class CNN_MLPClassifier(nn.Module):
    def __init__(self, sequence_length, 
                 kernel_size1=15, kernel_size2=15, 
                 num_filters1=32, num_filters2=64, 
                 hidden_dim=128):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=num_filters1, kernel_size=kernel_size1, padding=kernel_size1 // 2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(num_filters1, num_filters2, kernel_size=kernel_size2, padding=kernel_size2 // 2)

        conv_output_length = sequence_length // 4  # 2x max pooling

        self.fc1 = nn.Linear(num_filters2 * conv_output_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1, 4)  # (batch, seq_len, 4)
        x = x.permute(0, 2, 1)        # (batch, 4, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(1)

# ---------- Helper Functions----------
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
    fasta_chroms = set(fasta.keys())
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
            data.append(((encoded, chrom, start, end, strand, seq), label))
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

def save_predictions(output_path, meta, y_true, y_scores):
    from collections import defaultdict
    grouped = defaultdict(list)
    flat_rows = []

    for (chrom, start, end, strand, seq), true, score in zip(meta, y_true, y_scores):
        pred = 1 if score > 0.5 else 0
        if true == 1 and pred == 1:
            category = "True Positive"
        elif true == 0 and pred == 0:
            category = "True Negative"
        elif true == 0 and pred == 1:
            category = "False Positive"
        else:
            category = "False Negative"

        row = {
            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": strand,
            "sequence": seq,
            "true_label": true,
            "score": score,
            "prediction": pred,
            "category": category
        }

        grouped[category].append(row)
        flat_rows.append(row)

    # Write human-readable output with section headers
    with open(output_path, 'w') as f:
        for category in ["True Positive", "True Negative", "False Positive", "False Negative"]:
            entries = grouped.get(category, [])
            if not entries:
                continue
            f.write(f"### {category} ###\n")
            df = pd.DataFrame(entries)
            df.to_csv(f, index=False)
            f.write("\n")

    # Write machine-readable flat output
    flat_output_path = output_path.replace(".csv", "_flat.csv")
    pd.DataFrame(flat_rows).to_csv(flat_output_path, index=False)
    print(f"Saved machine-readable predictions to {flat_output_path}")

# ------- Main ---------
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
    all_data = [((pad_or_truncate(x[0], INPUT_DIM), x[1], x[2], x[3], x[4], x[5]), y) for x, y in labeled_data]

    if args.shuffle_labels:
        print("Shuffling labels for sanity check...")
        all_data = shuffle_labels(all_data)

    train_data, test_data = train_test_split(all_data, test_size=0.3, stratify=[y for _, y in all_data])
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    train_loader = DataLoader(SequenceDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SequenceDataset(test_data), batch_size=BATCH_SIZE)

    model = CNN_MLPClassifier(SEQUENCE_LENGTH)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                val_output = model(X_val)
                val_loss += criterion(val_output, y_val).item()

        print(f"Epoch {epoch + 1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")


    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating", leave=False):
            output = model(X)
            y_true.extend(y.numpy())
            y_scores.extend(output.numpy())

    meta = [(x[1], x[2], x[3], x[4], x[5]) for x, _ in test_data]
    y_pred = [1 if s > 0.5 else 0 for s in y_scores]

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

    log_metrics(args.log_file, roc, cm, report)

    if args.pred_output:
        save_predictions(args.pred_output, meta, y_true, y_scores)

if __name__ == "__main__":
    main()
