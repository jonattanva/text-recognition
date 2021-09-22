import unittest
import tensorflow as tf

from recognition.model.darknet.layer.convolutional import Convolutional


class ConvolutionalTestCase(unittest.TestCase):

    def test_name(self):
        convolutional = Convolutional(32)
        self.assertEqual('convolutional', convolutional.name)

        convolutional = Convolutional(32, name='convolutional_test')
        self.assertEqual('convolutional_test', convolutional.name)

    def test_model_shape(self):
        inputs = tf.zeros([1, 416, 416, 3])

        convolutional = Convolutional(32)
        inputs = convolutional(inputs, training=False)
        self.assertEqual([1, 416, 416, 32], inputs.shape)

        convolutional = Convolutional(64, use_bias=False, use_zero_padding=True)
        inputs = convolutional(inputs, training=False)
        self.assertEqual([1, 208, 208, 64], inputs.shape)

        convolutional = Convolutional(64, use_bias=False, use_zero_padding=True)
        inputs = convolutional(inputs, training=True)
        self.assertEqual([1, 104, 104, 64], inputs.shape)

    def test_weights(self):
        inputs = tf.zeros([1, 416, 416, 3])

        convolutional = Convolutional(64)
        convolutional(inputs, training=True)

        self.assertEqual(5, len(convolutional.weights))
        self.assertEqual(3, len(convolutional.trainable_weights))
        self.assertEqual(2, len(convolutional.non_trainable_weights))

        convolutional = Convolutional(32)
        convolutional(inputs, training=False)

        self.assertEqual(5, len(convolutional.weights))
        self.assertEqual(3, len(convolutional.trainable_weights))
        self.assertEqual(2, len(convolutional.non_trainable_weights))

    def test_config(self):
        convolutional = Convolutional(64)
        config = convolutional.get_config()
        self.assertEqual({
            'name': 'convolutional',
            'trainable': True,
            'dtype': 'float32',
            'filters': 64,
            'kernel_size': (3, 3),
            'use_bias': True,
            'use_zero_padding': False
        }, config)


if __name__ == '__main__':
    unittest.main()
