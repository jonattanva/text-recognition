import unittest
import PIL.Image
import tensorflow as tf
import recognition.util.image

from recognition.parse.tokenizer import Tokenizer
from recognition.service.classification import Classification


class ClassificationTestCase(unittest.TestCase):

    def setUp(self):
        self._classification = Classification()

    def test_loss_step(self):
        tokenizer = Tokenizer(one_hot=False, channel=3)
        image = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))

        label = 'hello world'
        content = recognition.util.image.to_bytes(image)
        serialize = tokenizer.serialize(content, label, size=(image.height, image.width), bbox_max_length=len(label))

        image, label = tokenizer.deserialize(serialize)
        predictions = self._classification._model(image, training=True)

        total_loss = self._classification.loss_step(label, predictions)
        self.assertEqual([1], total_loss.shape)

    def test_accuracy_step(self):
        tokenizer = Tokenizer(one_hot=False, channel=3)
        image = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))

        label = 'bye world'
        content = recognition.util.image.to_bytes(image)
        serialize = tokenizer.serialize(content, label, size=(image.height, image.width), bbox_max_length=len(label))

        image, label = tokenizer.deserialize(serialize)
        predictions = self._classification._model(image, training=True)

        labels, predictions = self._classification.accuracy_step(label, predictions)
        self.assertEqual(tf.int32, labels.dtype)
        self.assertEqual(tf.int32, predictions.dtype)


if __name__ == '__main__':
    unittest.main()
