import pandas as pd
import keras
import tensorflow as tf
from Bio import SeqIO
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import Row

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
                           vocab_embed:keras.layers.StringLookup) -> SparkSession:
    """
    Ingests refseq genome from FASTA, passes each entry to spark. Returns a spark session for accessing the data
    """
    spark = SparkSession.builder.getOrCreate()
    refseq_dfs = []
    print("=== Reading Genomic Data ===")
    with open(file_path, 'r') as file:
        for i, seqrec in enumerate(SeqIO.parse(file, 'fasta')):
            # read sequence
            sequence = str(seqrec.seq)
            # make into list for fancy matrix dicing
            sequence = list(sequence)
            array = []
            print(f"-> Processing {seqrec.id}")
            for i in range(kmer_size):
                new_slice = sequence[i::kmer_size]
                array.append(new_slice)
            array = np.array(array)
            array = array.T
            # store kmers in spark dataframe
            print(f"Exporting {seqrec.id} to spark dataframe")
            n_rows = [Row(id=seqrec.id, description=seqrec.description, kmer=''.join(r)) for r in array]

            df = spark.createDataFrame([n_rows]) # row = dataframe to get around 2gb serialization limit        
            refseq_dfs.append(df)
            # print progress
            if i % 10 == 0:
                print(f"-> {i} records loaded")
    
    print("=== RefSeq Genome Loaded ===")

    return spark, refseq_dfs, vocab_embed
    

def construct_tf_dataset(contigs: np.array) -> tf.data.Dataset:
    """
    Smooths out the contigs into a tf.data.Dataset object for training the transformer model with zero padding.
    """
    pass