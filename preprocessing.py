import pandas as pd

# TODO: Use Biopython to load human reference genome from GenBank Flat File

def load_promoter_sequences(promoter_file):
    data = pd.read_csv(promoter_file, header=None)
    promoter_sequences = data[2]
    promoter_sequences = [''.join(seq.split('\t')) for seq in promoter_sequences]
    return promoter_sequences

def preprocess_promoter_sequences(promoter_sequences):
    in_sequences = []
    out_sequences = []
    whole_sequences = []
    for seq in promoter_sequences:
        # masking and data augmentation

        # tokenize
        seq = list(seq)
        # add start and end tokens
        seq = ['<s>'] + seq + ['</s>']
        in_sequences.append(seq[:-1])
        out_sequences.append(seq[1:])
        whole_sequences.append(seq)
    return in_sequences, out_sequences, whole_sequences
