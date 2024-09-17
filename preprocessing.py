import pandas as pd

def load_promoter_sequences(promoter_file):
    data = pd.read_csv(promoter_file, header=None)
    promoter_sequences = data[2]
    promoter_sequences = [''.join(seq.split('\t')) for seq in promoter_sequences]
    return promoter_sequences

def preprocess_promoter_sequences(promoter_sequences):
    train_sequences = []
    val_sequences = []
    for seq in promoter_sequences:
        # masking and data augmentation

        # tokenize
        seq = list(seq)
        # add start and end tokens
        seq = ['<s>'] + seq + ['</s>']
        train_sequences.append(seq[:-1])
        val_sequences.append(seq[1:])
    return train_sequences, val_sequences
