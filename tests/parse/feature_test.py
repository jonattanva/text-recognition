import uuid
import unittest
import numpy as np
import tensorflow as tf
import recognition.util.image

from PIL import Image
from recognition.parse.feature import Feature


class FeatureTestCase(unittest.TestCase):

    def setUp(self):
        self.feature = Feature()

    def test_serialize_and_deserialize_options(self):
        image = Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        content = recognition.util.image.to_bytes(image)

        serialize = self.feature.serialize(
            content, b'label', size=(image.height, image.width),
            bbox=[[105.5, 46.5, 470.5, 121.5, 0], [117.5, 136.5, 454.5, 151.5, 0]],
            bbox_max_length=2)

        image, label, options = self.feature.deserialize(serialize)
        self.assertEqual((768, 576, 3), image.shape)
        self.assertEqual('label', label.numpy().decode('utf-8'))
        self.assertEqual(5, options[Feature.CLASSIFICATION_LABEL_MAX_LENGTH].numpy())
        self.assertEqual(2, options[Feature.LOCATION_BBOX_MAX_LENGTH])
        self.assertTrue(np.array_equal([[105.5, 46.5, 470.5, 121.5, 0.],
                                        [117.5, 136.5, 454.5, 151.5, 0.]], options[Feature.LOCATION_BBOX]))

    def test_serialize_and_deserialize(self):
        image = Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        content = recognition.util.image.to_bytes(image)

        serialize = self.feature.serialize(content, b'label', size=(image.height, image.width))

        image, label, options = self.feature.deserialize(serialize)
        self.assertEqual((768, 576, 3), image.shape)
        self.assertEqual('label', label.numpy().decode('utf-8'))
        self.assertEqual(5, options[Feature.CLASSIFICATION_LABEL_MAX_LENGTH].numpy())

    def test_float_feature(self):
        value = Feature.float_feature(float(1))
        self.assertTrue(value.HasField('float_list'))

        value = Feature.float_feature(1.0)
        self.assertTrue(value.HasField('float_list'))

        value = Feature.float_feature([1.0, 2.0])
        self.assertTrue(value.HasField('float_list'))

    def test_int64_feature(self):
        value = Feature.int64_feature(1)
        self.assertTrue(value.HasField('int64_list'))

        value = Feature.int64_feature(True)
        self.assertTrue(value.HasField('int64_list'))

        value = Feature.int64_feature([True, False])
        self.assertTrue(value.HasField('int64_list'))

    def test_bytes_feature(self):
        value = self.feature.bytes_feature(uuid.uuid1())
        self.assertTrue(value.HasField('bytes_list'))

        value = self.feature.bytes_feature('hi')
        self.assertTrue(value.HasField('bytes_list'))

    def test_padded_shapes(self):
        images, labels, extras = self.feature.padded_shapes()
        self.assertEqual([None, None, None], images)
        self.assertEqual([], labels)
        self.assertEqual({
            'location/bbox': [None, None],
            'location/bbox/length': [],
            'classification/label/length': [],
            'token': [],
            'image/height': [],
            'image/width': [],
            'version': [],
            'classification/label/alignment': []
        }, extras)

    def test_serialize(self):
        self.assertRaises(ValueError, self.feature.serialize, '', '', (0, 0))

    def test_int32_boxes(self):
        image = Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        content = recognition.util.image.to_bytes(image)

        serialize = self.feature.serialize(
            content, b'label', size=(image.height, image.width),
            bbox=[[105, 46, 470, 121, 0], [117, 136, 454, 151, 0]], bbox_max_length=2)

        image, label, options = self.feature.deserialize(serialize)
        self.assertEqual(tf.float32, options[Feature.LOCATION_BBOX].dtype)


if __name__ == '__main__':
    unittest.main()
