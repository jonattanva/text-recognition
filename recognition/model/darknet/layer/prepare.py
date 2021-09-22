# coding: utf-8
import tensorflow as tf

from recognition.model.darknet.layer.convolutional import Convolutional


class Prepare(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super(Prepare, self).__init__(name=kwargs.get('name', 'prepare'))
        self._filters = filters
        self._number_anchors = kwargs.get('number_anchors', 3)
        self._number_classes = kwargs.get('number_classes', 1)

        self._conv1 = Convolutional(filters=self._filters * 2, name="conv_1")
        self._conv2 = Convolutional(
            filters=self._number_anchors * (self._number_classes + 5),
            kernel_size=(1, 1),
            use_bias=False,
            name="conv_2")
        self._reshape = tf.keras.layers.Lambda(self.reshape)

    def build(self, input_shape):
        super(Prepare, self).build(input_shape)

    def get_config(self):
        config = super(Prepare, self).get_config()
        config.update({'filters': self._filters})
        config.update({'number_anchors': self._number_anchors})
        config.update({'number_classes': self._number_classes})
        return config

    def reshape(self, inputs):
        inputs_shape = tf.shape(inputs)
        return tf.reshape(
            inputs, shape=(-1, inputs_shape[1], inputs_shape[2], self._number_anchors, (self._number_classes + 5)))

    def call(self, inputs, training=False):
        inputs = self._conv1(inputs, training=training)
        inputs = self._conv2(inputs, training=training)
        inputs = self._reshape(inputs)
        return inputs
