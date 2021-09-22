import unittest
import tensorflow as tf
import recognition.util.path as path

from recognition.model.vision.vision import Vision


class VisionTestCase(unittest.TestCase):

    def setUp(self):
        self._chars = path.get_chars()

    def test_name(self):
        vision = Vision(self._chars)
        self.assertEqual('vision', vision.name)

    def test_shape(self):
        inputs = tf.zeros([5, 768, 576, 3])
        vision = Vision(self._chars)
        inputs = vision(inputs)
        self.assertEqual([24, 5, 107], inputs.shape)


if __name__ == '__main__':
    unittest.main()
