import unittest
import tensorflow as tf
import recognition.util.path as path

from recognition.model.darknet.yolov3 import YoloV3
from recognition.model.darknet.layer.boxes import Boxes


class BoxesTestCase(unittest.TestCase):

    def setUp(self):
        self._anchors = path.get_anchors()
        self._classes = path.get_class_name()

    def test_name(self):
        boxes = Boxes(anchors=self._anchors, anchors_masks=YoloV3.ANCHOR_MASKS)
        self.assertEqual('boxes', boxes.name)

        boxes = Boxes(anchors=self._anchors, anchors_masks=YoloV3.ANCHOR_MASKS, name='boxes_test')
        self.assertEqual('boxes_test', boxes.name)

    def test_shape(self):
        inputs = tf.zeros([48, 52, 52, 128])
        boxes = Boxes(
            anchors=self._anchors, anchors_masks=YoloV3.ANCHOR_MASKS, number_classes=len(self._classes))
        results = boxes(inputs=(inputs, inputs, inputs), training=True)
        self.assertEqual(3, len(results))

    def test_weights(self):
        inputs = tf.zeros([48, 52, 52, 128])
        boxes = Boxes(
            anchors=self._anchors, anchors_masks=YoloV3.ANCHOR_MASKS, number_classes=len(self._classes))
        boxes(inputs=(inputs, inputs, inputs), training=True)

        self.assertEqual(0, len(boxes.weights))
        self.assertEqual(0, len(boxes.trainable_weights))
        self.assertEqual(0, len(boxes.non_trainable_weights))


if __name__ == '__main__':
    unittest.main()
