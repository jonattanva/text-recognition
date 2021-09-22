# coding: utf-8
import tensorflow as tf

from recognition.model.darknet.layer.residual import Residual
from recognition.model.darknet.layer.convolutional import Convolutional


class Darknet53(tf.keras.layers.Layer):

    def __init__(self):
        super(Darknet53, self).__init__(name='darknet53')
        self._conv1 = Convolutional(filters=32, use_zero_padding=False, name='conv_1')
        self._conv2 = Convolutional(filters=64, use_zero_padding=True, name='conv_2')
        self._conv3 = Convolutional(filters=128, use_zero_padding=True, name='conv_3')
        self._conv4 = Convolutional(filters=256, use_zero_padding=True, name='conv_4')
        self._conv5 = Convolutional(filters=512, use_zero_padding=True, name='conv_5')
        self._conv6 = Convolutional(filters=1024, use_zero_padding=True, name='conv_6')

        self._resi1 = Residual(filters=(32, 64), name='resi_1')
        self._resi2 = Residual(filters=(64, 128), name='resi_2')
        self._resi3 = Residual(filters=(128, 256), name='resi_3')
        self._resi4 = Residual(filters=(256, 512), name='resi_4')
        self._resi5 = Residual(filters=(512, 1024), name='resi_5')

    def build(self, input_shape):
        super(Darknet53, self).build(input_shape)

    def call(self, inputs, training=False):
        inputs = self._conv1(inputs, training=training)
        inputs = self._conv2(inputs, training=training)
        inputs = self._resi1(inputs, training=training)

        inputs = self._conv3(inputs, training=training)
        for i in range(2):
            inputs = self._resi2(inputs, training=training)

        inputs = self._conv4(inputs, training=training)
        for i in range(8):
            inputs = self._resi3(inputs, training=training)

        inputs_a = inputs
        inputs = self._conv5(inputs, training=training)
        for i in range(8):
            inputs = self._resi4(inputs, training=training)

        inputs_b = inputs
        inputs = self._conv6(inputs, training=training)
        for i in range(4):
            inputs = self._resi5(inputs, training=training)

        return inputs_a, inputs_b, inputs
