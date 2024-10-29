import tensorflow as tf
import keras

from preprocessing import (
    load_promoter_sequences, 
    preprocess_promoter_sequences, 
    ingest_refseq_genome, 
    construct_tf_dataset
)
from transformer import Transformer


# load and preprocess the data
spark = ingest_refseq_genome(
  file_path='./data/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna',
  kmer_size=3,
)

# batch process kmer chunks into varying length sequences
#contig_size_range=(500, 1000),
#  vocab_embed=keras.layers.StringLookup(
#    oov_token='[UNK]',
#    mask_token='[MASK]',
#    output_mode='one_hot'
#  ),
exit(0)

#sequences = load_promoter_sequences('./data/promoters.data')
#in_tensor, out_tensor, whole_tensor = preprocess_promoter_sequences(sequences)
#in_tensor = tf.pad(in_tensor, [[0, 0], [0, 1],[0,0]])
#out_tensor = tf.pad(out_tensor, [[0, 0], [0, 1],[0,0]])

#inputs = (whole_tensor, in_tensor)
#labels = out_tensor
#ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
#print(ds)

# vocabulary
#initial_embedder = keras.layers.StringLookup(
#    oov_token='[UNK]',
#    mask_token='[MASK]',
#    output_mode='one_hot'
#)

# initialize the model
num_layers = 8
d_model = 128
num_heads = 8
dff = 1024
input_vocab_size = len(initial_embedder.get_vocabulary())
target_vocab_size = len(initial_embedder.get_vocabulary())
dropout_rate = 0.2

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    dropout_rate=dropout_rate
)

# initialize the optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# initialize the loss and accuracy metrics
def masked_loss(label, pred):
  mask = label != 0
  loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

# compile the model
print("=== COMPILING ===")
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])
print("DONE")

# train the model
print(transformer.summary())
tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.001,
    patience=5, 
    restore_best_weights=True
    )
history = transformer.fit(ds, epochs=1000, callbacks=[tb_callback, early_stopping])
transformer.save('./models/promoter_only')

tb.notebook