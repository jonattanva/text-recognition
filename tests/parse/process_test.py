import unittest
import PIL.Image
import recognition.util.image
import recognition.util.path as path

from recognition.model.darknet.yolov3 import YoloV3
from recognition.parse.process import Process


class ProcessTestCase(unittest.TestCase):

    def setUp(self):
        self._transform = Process(anchors=path.get_anchors(), channel=3)

    def test_serialize_and_deserialize(self):
        image = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))

        content = recognition.util.image.to_bytes(image)
        serialize = self._transform.serialize(
            content, b'label', size=(image.height, image.width),
            bbox=[[105.5, 46.5, 470.5, 121.5, 0], [117.5, 136.5, 454.5, 151.5, 0]], bbox_max_length=2)

        image, bbox = self._transform.deserialize(serialize)
        self.assertEqual((416, 416, 3), image.shape)
        self.assertEqual((1, 13, 13, 3, 6), bbox[0].shape)
        self.assertEqual((1, 26, 26, 3, 6), bbox[1].shape)
        self.assertEqual((1, 52, 52, 3, 6), bbox[2].shape)

    def test_padded_shapes(self):
        self.assertIsNone(self._transform.padded_shapes())

    def test_convert(self):
        result = Process.convert([[[10, 10, 40, 40, 1, 1]]], 13, YoloV3.ANCHOR_MASKS[0])
        self.assertEqual([1, 13, 13, 3, 6], result.shape)

        result = Process.convert([[[10, 10, 40, 40, 1, 1]]], 26, YoloV3.ANCHOR_MASKS[1])
        self.assertEqual([1, 26, 26, 3, 6], result.shape)

        result = Process.convert([[[10, 10, 40, 40, 1, 1]]], 52, YoloV3.ANCHOR_MASKS[1])
        self.assertEqual([1, 52, 52, 3, 6], result.shape)


if __name__ == '__main__':
    unittest.main()
