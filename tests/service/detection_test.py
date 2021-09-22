import unittest
import PIL.Image
import tensorflow as tf
import recognition.util.image
import recognition.util.path as path

from recognition.parse.process import Process
from recognition.service.detection import Detection


class DetectionTestCase(unittest.TestCase):

    def setUp(self):
        self._anchors = path.get_anchors()
        self._classes = path.get_class_name()
        self._detection = Detection(self._anchors, self._classes)

    def test_loss_step(self):
        transform = Process(anchors=self._anchors, channel=3)
        image = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))

        content = recognition.util.image.to_bytes(image)
        serialize = transform.serialize(
            content, b'label', size=(image.height, image.width), bbox=[[0, 0, 40, 20, 1]],
            bbox_max_length=1)

        image, boxes = transform.deserialize(serialize)
        predictions = self._detection._model(image, training=True)

        total_loss = self._detection.loss_step(boxes, predictions)
        self.assertTrue(total_loss > 0)
        self.assertEqual(tf.float32, total_loss.dtype)

    def test_accuracy_step(self):
        transform = Process(anchors=self._anchors, channel=3)
        image = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))

        content = recognition.util.image.to_bytes(image)
        serialize = transform.serialize(
            content, b'label', size=(image.height, image.width),
            bbox=[[0, 0, 40, 20, 1]], bbox_max_length=1)

        image, boxes = transform.deserialize(serialize)
        predictions = self._detection._model(image, training=True)

        accuracy_value = self._detection.accuracy_step(boxes, predictions)
        self.assertTrue(accuracy_value > 0)


if __name__ == '__main__':
    unittest.main()
