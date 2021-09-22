import unittest
import tensorflow as tf

from recognition.model.vision.layer.convolutional import Convolutional


class ConvolutionalTestCase(unittest.TestCase):

    def test_name(self):
        convolutional = Convolutional(64)
        self.assertEqual('convolutional', convolutional.name)

        convolutional = Convolutional(64, name='convolutional_test')
        self.assertEqual('convolutional_test', convolutional.name)

    def test_input_shape(self):
        inputs = tf.zeros([768, 576, 3])
        convolutional = Convolutional(32, input_shape=(768, 576, 3))
        inputs = convolutional(inputs, training=True)
        self.assertEqual([1, 384, 288, 32], inputs.shape)

        inputs = tf.zeros([768, 576, 3])
        convolutional = Convolutional(32)
        inputs = convolutional(inputs, training=True)
        self.assertEqual([1, 384, 288, 32], inputs.shape)

    def test_shape(self):
        inputs = tf.zeros([768, 576, 3])
        convolutional = Convolutional(32)
        inputs = convolutional(inputs, training=True)
        self.assertEqual([1, 384, 288, 32], inputs.shape)

        convolutional = Convolutional(64)
        results = convolutional(inputs, training=True)
        self.assertEqual([1, 192, 144, 64], results.shape)

        inputs = tf.zeros([1, 768, 576, 3])
        convolutional = Convolutional(32)
        inputs = convolutional(inputs, training=True)

        convolutional = Convolutional(64)
        inputs = convolutional(inputs, training=True)

        convolutional = Convolutional(128, kernel_size=(3, 3))
        inputs = convolutional(inputs, training=True)

        convolutional = Convolutional(128, kernel_size=(3, 3))
        inputs = convolutional(inputs, training=True)

        convolutional = Convolutional(256, kernel_size=(3, 3))
        inputs = convolutional(inputs, training=True)

        self.assertEqual([1, 24, 18, 256], inputs.shape)


if __name__ == '__main__':
    unittest.main()
