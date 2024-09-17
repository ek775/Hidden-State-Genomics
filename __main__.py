import tensorflow as tf
import tensorboard as tb

from preprocessing import load_promoter_sequences, preprocess_promoter_sequences

sequences = load_promoter_sequences('./data/promoters.data')
train_sequences, val_sequences = preprocess_promoter_sequences(sequences)


