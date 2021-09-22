import unittest
import tensorflow as tf
import recognition.util.path as path

from recognition.model.darknet.yolov3 import YoloV3


class YoloV3TestCase(unittest.TestCase):

    def setUp(self):
        self.classes = path.get_class_name()
        self.anchors = path.get_anchors()

    def test_reshape(self):
        inputs = [
            tf.ones((1, 13, 13, 3, 4)),
            tf.ones((1, 13, 13, 3, 1)),
            tf.ones((1, 13, 13, 3, 3))
        ]

        result = YoloV3.reshape(inputs, index=0)
        self.assertEqual([1, 507, 4], result.shape)

        result = YoloV3.reshape(inputs, index=1)
        self.assertEqual([1, 507, 1], result.shape)

        result = YoloV3.reshape(inputs, index=2)
        self.assertEqual([1, 507, 3], result.shape)

    def test_combined_bounding_boxes(self):
        images = tf.zeros([416, 416, 3])
        yolo = YoloV3(classes=self.classes, anchors=self.anchors)

        outputs = yolo(images, training=False)
        boxes, scores, classes, prediction_box = YoloV3.combined_bounding_boxes(outputs)
        self.assertEqual((1, 100, 4), boxes.shape)
        self.assertEqual((1, 100), scores.shape)
        self.assertEqual((1, 100), classes.shape)
        self.assertEqual((1,), prediction_box.shape)

    def test_model_shape(self):
        images = tf.zeros([416, 416, 3])
        yolo = YoloV3(classes=self.classes, anchors=self.anchors)

        inputs = yolo(images, training=True)
        self.assertEqual((1, 13, 13, 3, 8), inputs[0].shape)
        self.assertEqual((1, 26, 26, 3, 8), inputs[1].shape)
        self.assertEqual((1, 52, 52, 3, 8), inputs[2].shape)

        inputs = yolo(images, training=False)
        self.assertEqual((1, 13, 13, 3, 4), inputs[0][0].shape)
        self.assertEqual((1, 13, 13, 3, 1), inputs[0][1].shape)
        self.assertEqual((1, 13, 13, 3, 3), inputs[0][2].shape)

        self.assertEqual((1, 26, 26, 3, 4), inputs[1][0].shape)
        self.assertEqual((1, 26, 26, 3, 1), inputs[1][1].shape)
        self.assertEqual((1, 26, 26, 3, 3), inputs[1][2].shape)

        self.assertEqual((1, 52, 52, 3, 4), inputs[2][0].shape)
        self.assertEqual((1, 52, 52, 3, 1), inputs[2][1].shape)
        self.assertEqual((1, 52, 52, 3, 3), inputs[2][2].shape)

    def test_save_and_load_weight(self):
        inputs = tf.zeros([416, 416, 3])
        yolo = YoloV3(classes=self.classes, anchors=self.anchors)
        yolo(inputs, training=True)

        filepath = path.resolve('temp/model/{}.h5'.format(yolo.name))
        yolo.save_weights(filepath)
        self.assertTrue(path.exists(filepath))

        yolo.load_weights(filepath)
        weights = yolo.get_weights()
        self.assertTrue(len(weights) > 0)

    def test_load_weight(self):
        yolo = YoloV3(classes=self.classes, anchors=self.anchors)
        yolo.build((416, 416, 3))
        yolo.load_weights()

        weights = yolo.get_weights()
        self.assertTrue(len(weights) > 0)

        inputs = tf.zeros([416, 416, 3])
        result = yolo(inputs, training=False)
        self.assertTrue(len(result) > 0)


if __name__ == '__main__':
    unittest.main()
