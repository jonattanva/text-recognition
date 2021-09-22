import unittest
import tensorflow as tf

from recognition.model.vision.backbone.recognition import Recognition


class ExtractTestCase(unittest.TestCase):

    def test_name(self):
        recognition = Recognition()
        self.assertEqual('recognition', recognition.name)

        recognition = Recognition(name='recognition_temp')
        self.assertEqual('recognition_temp', recognition.name)

    def test_shape(self):
        inputs = tf.zeros([1, 768, 576, 3])

        recognition = Recognition()
        inputs = recognition(inputs, training=True)
        self.assertEqual([1, 24, 512], inputs.shape)


if __name__ == '__main__':
    unittest.main()
