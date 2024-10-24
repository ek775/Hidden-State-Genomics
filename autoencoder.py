import tensorflow as tf
import keras
from keras import layers
import numpy as np
from tensorboard import program
import Bio
import transformers

# create dataset from random tensors to test
SAE_name = 'autoencoder.test'
embed_length = 2048
ef = 4

print("=== Generating Test Data ===")
fake_embeddings = tf.random.uniform(shape=[1000, embed_length])
fake_dataset = tf.data.Dataset.from_tensor_slices((fake_embeddings, fake_embeddings)).batch(100)

# TODO: load ESM-2 and NTv2 models + data + tokenizers from hugging face

# define our autoencoder in encoder / decoder components
class Encoder(layers.Layer):
    # L1 gives us sparsity
    # project model activations into high-dimensional space
    def __init__(self, encoding_size:int, n_features:int):
        super().__init__()
        self.input_layer = layers.InputLayer(shape=(encoding_size,))
        self.features = layers.Dense(units=n_features, kernel_regularizer='l1', activity_regularizer='l1', activation='relu')

    def call(self, x):
        x = self.features(x)
        return x

class Decoder(layers.Layer):
    # trained weights here give us the "feature directions"
    def __init__(self, n_features:int, encoding_size:int):
        super().__init__()
        self.input_layer = layers.Input(shape=(n_features,))
        self.output_layer = layers.Dense(units=encoding_size, activation='relu')

    def call(self, x):
        x = self.output_layer(x)
        return x

class SparseAutoEncoder(tf.keras.Model):
    def __init__(self, encoding_size:int, expansion_factor:int):
        super().__init__()
        self.n_features = encoding_size * expansion_factor
        self.encoder = Encoder(encoding_size=encoding_size, n_features=self.n_features)
        self.decoder = Decoder(n_features=self.n_features, encoding_size=encoding_size)

    def call(self, x):
        x = self.encoder(x)
        # cache feature activations for mapping
        self.last_features = x
        # decode back to original space
        x = self.decoder(x)
        return x

print("=== Initializing Model ===")
# initialize the optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# initialize the loss function
loss = keras.losses.MeanSquaredError()

# initialize the metrics
metrics = [
    keras.metrics.AUC(from_logits=True),
    keras.metrics.CosineSimilarity(),
    keras.metrics.MeanSquaredError(),
]

# compile the model
autoencoder = SparseAutoEncoder(encoding_size=embed_length, expansion_factor=ef)
autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print(autoencoder.summary())

tb_callback = keras.callbacks.TensorBoard(log_dir=f'./logs/{SAE_name}', histogram_freq=5)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='mean_squared_error',
    min_delta=0.001,
    patience=10, 
    restore_best_weights=True
    )

print("=== Training Model ===")
history = autoencoder.fit(fake_dataset, epochs=100, callbacks=[tb_callback, early_stopping])
print("=== Saving Model ===")
autoencoder.save(f'./models/{SAE_name}.keras')
print("Displaying Results in Tensorboard...")
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', './logs'])
url = tb.launch()
print(f"TensorBoard started at {url}")