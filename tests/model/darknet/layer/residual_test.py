import unittest
import tensorflow as tf

from recognition.model.darknet.layer.residual import Residual


class ResidualTestCase(unittest.TestCase):

    def test_name(self):
        residual = Residual((32, 64))
        self.assertEqual('residual', residual.name)

        residual = Residual((32, 64), name='residual_test')
        self.assertEqual('residual_test', residual.name)

    def test_model_shape(self):
        inputs = tf.ones([48, 208, 208, 64])
        residual = Residual((32, 64))
        result = residual(inputs, training=False)
        self.assertEqual([48, 208, 208, 64], result.shape)

        inputs = tf.ones([48, 208, 208, 32])
        residual = Residual((16, 32))
        result = residual(inputs, training=True)
        self.assertEqual([48, 208, 208, 32], result.shape)

    def test_weights(self):
        inputs = tf.ones([48, 208, 208, 64])

        residual = Residual((32, 64))
        residual(inputs, training=True)

        self.assertEqual(10, len(residual.weights))
        self.assertEqual(6, len(residual.trainable_weights))
        self.assertEqual(4, len(residual.non_trainable_weights))

    def test_config(self):
        residual = Residual((32, 64))
        config = residual.get_config()
        self.assertEqual({
            'dtype': 'float32',
            'filters': (32, 64),
            'name': 'residual',
            'trainable': True
        }, config)


if __name__ == '__main__':
    unittest.main()
