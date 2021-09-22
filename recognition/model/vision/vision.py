# coding: utf-8
import tensorflow as tf

from recognition.model.model import Model
from recognition.model.vision.backbone.recognition import Recognition
from recognition.model.darknet.layer.normalization import Normalization


class Vision(Model):

    def __init__(self, chars):
        super(Vision, self).__init__(name='vision')
        self._chars = chars
        self._number_chars = len(self._chars)

        self._recognition = Recognition()
        self._normalization = Normalization()
        self._dense = tf.keras.layers.Dense(self._number_chars + 1)
        self._softmax = tf.keras.layers.Softmax()

    def build(self, input_shape):
        super(Vision, self).build(input_shape)

    def call(self, inputs, training=False, mask=None):
        inputs_shape = inputs.get_shape()
        if inputs_shape.ndims == 3:
            inputs = tf.expand_dims(inputs, axis=0)

        inputs = self._recognition(inputs, training=training)
        inputs = self._normalization(inputs, training=training)
        inputs = self._dense(inputs)
        inputs = self._softmax(inputs)

        inputs_shape = inputs.get_shape()
        if inputs_shape.ndims == 3:
            inputs = tf.transpose(inputs, perm=[1, 0, 2])

        return inputs
