import unittest
import tensorflow as tf

from recognition.model.darknet.layer.normalization import Normalization


class NormalizationTestCase(unittest.TestCase):

    def test_name(self):
        normalization = Normalization()
        self.assertEqual('normalization', normalization.name)

        normalization = Normalization(name='normalization_test')
        self.assertEqual('normalization_test', normalization.name)

    def test_shape(self):
        inputs = tf.zeros([100, 416, 416, 3])
        normalization = Normalization()
        results = normalization(inputs, training=False)
        self.assertEqual([100, 416, 416, 3], results.shape)

        inputs = tf.zeros([100, 416, 416, 3])
        normalization = Normalization()
        results = normalization(inputs, training=True)
        self.assertEqual([100, 416, 416, 3], results.shape)


if __name__ == '__main__':
    unittest.main()
