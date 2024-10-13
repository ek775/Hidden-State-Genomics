import pandas as pd
import keras
import tensorflow as tf
from Bio import SeqIO
import numpy as np

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

    # convert to one-hot encoding
    encoding = keras.layers.StringLookup(
        oov_token='[UNK]',
        output_mode='one_hot'
        )
    encoding.adapt(whole_sequences)
    in_sequences = encoding(in_sequences)
    out_sequences = encoding(out_sequences)
    whole_sequences = encoding(whole_sequences)

    return in_sequences, out_sequences, whole_sequences


def ingest_refseq_genome(file_path:str, 
                           kmer_size:int, 
                           contig_size_range:tuple, 
                           vocab_embed:keras.layers.StringLookup) -> np.array:
    """
    Ingests refseq genome from FASTA, batch processes it into kmers, and generates contigs for training.
    """
    tokens = []
    print("=== Reading Genomic Data ===")
    with open(file_path, 'r') as file:
        for i, seqrec in enumerate(SeqIO.parse(file, 'fasta')):
            # tokenize each fasta entry
            sequence = list(str(seqrec.seq))
            for nt_index in range(0, len(sequence), kmer_size):
                kmer = sequence[nt_index:nt_index+kmer_size]
                tokens.append(kmer)                
            # print progress
            if i % 10 == 0:
                print(f"-> {i} records loaded")
    print("=== RefSeq Genome Loaded ===")
    print(f"Total Tokens: {len(tokens)}")

    return tokens
    

def construct_tf_dataset(contigs: np.array) -> tf.data.Dataset:
    """
    Smooths out the contigs into a tf.data.Dataset object for training the transformer model with zero padding.
    """
    pass