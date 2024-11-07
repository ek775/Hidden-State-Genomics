import tensorflow as tf
import keras
from keras import layers
import numpy as np
import Bio
import transformers

# create dataset from random tensors to test
SAE_name = 'autoencoder_test'
embed_length = 2048
ef = 4

print("=== Generating Test Data ===")
fake_embeddings = tf.random.uniform(shape=[1000, embed_length])
fake_dataset = tf.data.Dataset.from_tensor_slices((fake_embeddings, fake_embeddings)).batch(100)

# TODO: load ESM-2 and NTv2 models + data + tokenizers from hugging face

@keras.saving.register_keras_serializable(package='SAE', name='SparseAutoEncoder')
class SparseAutoEncoder(keras.Model):
    def __init__(self, encoding_size:int, expansion_factor:int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = encoding_size * expansion_factor
        self.input_layer = layers.InputLayer(shape=(encoding_size,))
        self.encoder = layers.Dense(
            units=self.n_features,
            activation='relu',
            kernel_regularizer='l1',
            activity_regularizer='l1'
        )
        self.decoder = layers.Dense(units=encoding_size, activation='relu')
        self.last_features = None
    
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_layer":self.input_layer,
                "encoder":self.encoder,
                "decoder":self.decoder,
                "n_features":self.n_features,
                "encoding_size":self.decoder.units,
                "expansion_factor":self.n_features/self.decoder.units
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def build(self, input_shape):
        self.add_weight(
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True
        )

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
path = f'./models/{SAE_name}.keras'
#autoencoder.save(f'./models/{SAE_name}.keras')
keras.models.save_model(autoencoder, path)
print(f"Model saved to: {path}")
print(autoencoder.summary())