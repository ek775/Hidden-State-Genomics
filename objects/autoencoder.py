import tensorflow as tf
import keras
from keras import layers

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
        return x, self.last_features