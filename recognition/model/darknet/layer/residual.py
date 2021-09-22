# coding: utf-8
import tensorflow as tf

from recognition.model.darknet.layer.convolutional import Convolutional


class Residual(tf.keras.layers.Layer):

    def __init__(self, filters, name='residual'):
        super(Residual, self).__init__(name=name)
        self._filters = filters
        self._conv1 = Convolutional(filters=self._filters[0], kernel_size=(1, 1), name='conv_1')
        self._conv2 = Convolutional(filters=self._filters[1], kernel_size=(3, 3), name='conv_2')
        self._add = tf.keras.layers.Add()

    def build(self, input_shape):
        super(Residual, self).build(input_shape)

    def get_config(self):
        config = super(Residual, self).get_config()
        config.update({'filters': self._filters})
        return config

    def call(self, inputs, training=False):
        previous = inputs
        inputs = self._conv1(inputs, training=training)
        inputs = self._conv2(inputs, training=training)
        inputs = self._add([previous, inputs])
        return inputs
