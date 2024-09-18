import tensorflow as tf
import tensorboard as tb

from preprocessing import load_promoter_sequences, preprocess_promoter_sequences
from model import Transformer


# load and preprocess the data
sequences = load_promoter_sequences('./data/promoters.data')
train_sequences, val_sequences = preprocess_promoter_sequences(sequences)


# initialize the model
transformer = Transformer(
    num_layers=8,
    d_model=128,
    num_heads=8,
    dff=1024,
    input_vocab_size=6,
    target_vocab_size=6,
    dropout_rate=0.2,
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
