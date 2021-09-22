# coding: utf-8
import tensorflow as tf

from recognition.model.darknet.layer.normalization import Normalization


class Convolutional(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super(Convolutional, self).__init__(name=kwargs.get('name', 'convolutional'))
        self._filters = filters
        self._kernel_size = kwargs.get('kernel_size', (3, 3))
        self._use_bias = kwargs.get('use_bias', True)
        self._use_zero_padding = kwargs.get('use_zero_padding', False)

        self._zero_padding = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))
        self._conv2D = tf.keras.layers.Conv2D(
            filters=self._filters,
            use_bias=not self._use_bias,
            kernel_size=self._kernel_size,
            strides=(2, 2) if self._use_zero_padding else (1, 1),
            padding="valid" if self._use_zero_padding else "same",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005))

        self._normalization = Normalization()
        self._leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def build(self, input_shape):
        super(Convolutional, self).build(input_shape)

    def get_config(self):
        config = super(Convolutional, self).get_config()
        config.update({'filters': self._filters})
        config.update({'kernel_size': self._kernel_size})
        config.update({'use_bias': self._use_bias})
        config.update({'use_zero_padding': self._use_zero_padding})
        return config

    def call(self, inputs, training=False):
        if self._use_zero_padding:
            inputs = self._zero_padding(inputs)

        inputs = self._conv2D(inputs)
        if self._use_bias:
            inputs = self._normalization(inputs, training=training)
            inputs = self._leaky_relu(inputs)

        return inputs
