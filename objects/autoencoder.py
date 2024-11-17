import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class SparseAutoEncoder(tf.keras.Model):
    def __init__(self, encoding_size:int, expansion_factor:int, name="sparse_auto_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_features = encoding_size * expansion_factor
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(encoding_size,))
        self.encoder = tf.keras.layers.Dense(
            units=self.n_features,
            use_bias=False,
            activation='relu',
            kernel_regularizer='l1',
            activity_regularizer='l1'
        )
        self.decoder = tf.keras.layers.Dense(units=encoding_size, use_bias=False, activation='relu')
        self.last_features = None

    def call(self, x):
        x = self.encoder(x)
        # cache feature activations for mapping
        self.last_features = x
        # decode back to original space
        x = self.decoder(x)
        return x, self.last_features
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'encoding_size': self.encoder.units,
            'expansion_factor': self.n_features // self.encoder.units
        })
        return config
    
    # default from_config & build methods