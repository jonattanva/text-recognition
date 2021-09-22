import tensorflow as tf


class Bidirectional(tf.keras.layers.Layer):

    def __init__(self, units=128, name="bidirectional"):
        super(Bidirectional, self).__init__(name=name)
        self._units = units
        self._lstm = tf.keras.layers.LSTM(
            units, return_sequences=True, kernel_initializer='he_normal')
        self._lstm_backwards = tf.keras.layers.LSTM(
            units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')
        self._reverse = tf.keras.layers.Lambda(
            lambda value: tf.keras.backend.reverse(value, axes=1))

    def build(self, input_shape):
        super(Bidirectional, self).build(input_shape)

    def get_config(self):
        config = super(Bidirectional, self).get_config()
        config.update({'units': self._units})
        return config

    def call(self, inputs, training=False, concatenate=False):
        before = self._lstm(inputs)
        inputs = self._lstm_backwards(inputs)
        inputs = self._reverse(inputs)
        inputs = tf.keras.layers.concatenate([before, inputs]) if concatenate \
            else tf.keras.layers.add([before, inputs])
        return inputs
