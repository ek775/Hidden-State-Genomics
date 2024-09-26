import tensorflow as tf
import keras
import tensorboard as tb

from preprocessing import load_promoter_sequences, preprocess_promoter_sequences, prepare_batch, make_batches
from model import Transformer


# load and preprocess the data
sequences = load_promoter_sequences('./data/promoters.data')
train_sequences, val_sequences = preprocess_promoter_sequences(sequences)
train_tensor = tf.convert_to_tensor(train_sequences)
val_tensor = tf.convert_to_tensor(val_sequences)
ds = tf.data.Dataset.from_tensor_slices([train_tensor, val_tensor])

# initialize the model
num_layers = 8
d_model = 128
num_heads = 8
dff = 1024
input_vocab_size = 6
target_vocab_size = 6
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
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])
