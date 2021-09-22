# coding: utf-8
import tensorflow as tf

from recognition.model.darknet.layer.normalization import Normalization


class Convolutional(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(5, 5), **kwargs):
        super(Convolutional, self).__init__(name=kwargs.get('name', 'convolutional'))
        self._filters = filters
        self._kernel_size = kernel_size
        self._input_shape = kwargs.get('input_shape', [])
        self._conv2D = tf.keras.layers.Conv2D(
            filters=self._filters,
            kernel_size=self._kernel_size,
            strides=(1, 1),
            padding='same',
            input_shape=self._input_shape)
        self._normalization = Normalization()
        self._softmax = tf.keras.layers.Softmax()
        self._max_pooling = tf.keras.layers.MaxPooling2D(padding='valid')

    def build(self, input_shape):
        super(Convolutional, self).build(input_shape)

    def get_config(self):
        config = super(Convolutional, self).get_config()
        config.update({'filters': self._filters})
        config.update({'kernel_size': self._kernel_size})
        config.update({'input_shape': self._input_shape})
        return config

    def call(self, inputs, training=False):
        inputs_shape = inputs.get_shape()
        if inputs_shape.ndims == 3:
            inputs = tf.expand_dims(inputs, axis=0)

        inputs = self._conv2D(inputs)
        inputs = self._normalization(inputs, training=training)
        inputs = self._softmax(inputs)
        inputs = self._max_pooling(inputs)
        return inputs
