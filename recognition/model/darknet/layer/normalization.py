# coding: utf-8
import tensorflow as tf


class Normalization(tf.keras.layers.BatchNormalization):
    """Normaliza las activaciones de la capa anterior por cada lote"""

    def __init__(self, name='normalization'):
        super(Normalization, self).__init__(name=name)

    def call(self, inputs, training=False):
        if not training:
            training = tf.constant(False, dtype=tf.bool)

        training = tf.logical_and(training, self.trainable)
        return super(Normalization, self).call(inputs, training=training)
