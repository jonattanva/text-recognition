import unittest
import PIL.Image
import numpy as np
import recognition.util.image

from recognition.parse.tokenizer import Tokenizer


class TokenizerTestCase(unittest.TestCase):

    def setUp(self):
        self._tokenizer = Tokenizer(channel=3)

    def test_padded_shapes(self):
        self.assertIsNone(self._tokenizer.padded_shapes())

    def test_serialize_and_deserialize(self):
        image = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))

        content = recognition.util.image.to_bytes(image)
        serialize = self._tokenizer.serialize(
            content, b'hello world', size=(image.height, image.width),
            bbox=[[105.5, 46.5, 470.5, 121.5, 0], [117.5, 136.5, 454.5, 151.5, 0]],
            bbox_max_length=2)

        image, label = self._tokenizer.deserialize(serialize)
        self.assertTrue(np.array_equal([[8, 5, 12, 12, 16, 0, 24, 16, 19, 12, 4]], label.numpy()))


if __name__ == '__main__':
    unittest.main()
