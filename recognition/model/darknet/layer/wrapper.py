# coding: utf-8
import tensorflow as tf

from recognition.model.darknet.layer.convolutional import Convolutional


class Wrapper(tf.keras.layers.Layer):

    def __init__(self, filters, name='wrapper'):
        super(Wrapper, self).__init__(name=name)
        self._filters = filters
        self._conv0 = Convolutional(filters=self._filters, kernel_size=(1, 1), name='conv_0')
        self._conv1 = Convolutional(filters=self._filters, kernel_size=(1, 1), name='conv_1')
        self._conv2 = Convolutional(filters=self._filters * 2, kernel_size=(3, 3), name='conv_2')
        self._conv3 = Convolutional(filters=self._filters, kernel_size=(1, 1), name='conv_3')
        self._conv4 = Convolutional(filters=self._filters * 2, kernel_size=(3, 3), name='conv_4')
        self._conv5 = Convolutional(filters=self._filters, kernel_size=(1, 1), name='conv_5')
        self._up_sampling = tf.keras.layers.UpSampling2D(size=(2, 2))
        self._concatenate = tf.keras.layers.Concatenate(axis=-1)

    def build(self, input_shape):
        super(Wrapper, self).build(input_shape)

    def get_config(self):
        config = super(Wrapper, self).get_config()
        config.update({'filters': self._filters})
        return config

    def call(self, inputs, training=False):
        if isinstance(inputs, tuple):
            inputs, inputs_skip = inputs[0], inputs[1]
            inputs = self._conv0(inputs, training=training)
            inputs = self._up_sampling(inputs)
            inputs = self._concatenate([inputs, inputs_skip])

        inputs = self._conv1(inputs, training=training)
        inputs = self._conv2(inputs, training=training)
        inputs = self._conv3(inputs, training=training)
        inputs = self._conv4(inputs, training=training)
        inputs = self._conv5(inputs, training=training)
        return inputs
