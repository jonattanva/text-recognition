# coding: utf-8
import tensorflow as tf

from recognition.model.vision.layer.bidirectional import Bidirectional
from recognition.model.vision.layer.convolutional import Convolutional


class Recognition(tf.keras.layers.Layer):
    DROPOUT_RATE = 0.2
    BIDIRECTIONAL_UNITS = 256

    def __init__(self, name='recognition'):
        super(Recognition, self).__init__(name=name)
        self._conv0 = Convolutional(32, kernel_size=(5, 5), name='conv_0')
        self._conv1 = Convolutional(64, kernel_size=(5, 5), name='conv_1')
        self._conv2 = Convolutional(128, kernel_size=(3, 3), name='conv_2')
        self._conv3 = Convolutional(128, kernel_size=(3, 3), name='conv_4')
        self._conv4 = Convolutional(256, kernel_size=(3, 3), name='conv_5')
        self._prepare = tf.keras.layers.Lambda(Recognition.prepare)
        self._bidi1 = Bidirectional(units=Recognition.BIDIRECTIONAL_UNITS, name="bidi_1")
        self._bidi2 = Bidirectional(units=Recognition.BIDIRECTIONAL_UNITS, name="bidi_2")
        self._dropout = tf.keras.layers.Dropout(Recognition.DROPOUT_RATE)

    def build(self, input_shape):
        super(Recognition, self).build(input_shape)

    @staticmethod
    def prepare(inputs):
        inputs_shape = inputs.get_shape()
        if inputs_shape.ndims > 3:
            batch_size, height, width, channel = inputs_shape
            inputs = tf.reshape(inputs, shape=(batch_size, height, width * channel))
        return inputs

    def call(self, inputs, training=False):
        inputs = self._conv0(inputs, training=training)
        inputs = self._conv1(inputs, training=training)
        inputs = self._conv2(inputs, training=training)
        inputs = self._conv3(inputs, training=training)
        inputs = self._conv4(inputs, training=training)
        inputs = self._prepare(inputs)
        inputs = self._bidi1(inputs, training=training, concatenate=False)
        inputs = self._bidi2(inputs, training=training, concatenate=True)
        inputs = self._dropout(inputs, training=training)
        return inputs
