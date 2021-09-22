import unittest
import PIL.Image
import PIL.ImageDraw
import numpy as np
import tensorflow as tf
import recognition.util.path as path
import recognition.util.image as image

from recognition.parse.feature import Feature


class ImageTestCase(unittest.TestCase):

    def setUp(self):
        self._feature = Feature()
        self._template = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))

    def test_resize_image(self):
        content = image.to_bytes(self._template)
        serialize = self._feature.serialize(content, b'label', size=(self._template.height, self._template.width))
        images, labels, extras = self._feature.deserialize(serialize)

        resize = image.resize(images, size=(416, 416))
        self.assertEqual((416, 416, 3), resize.shape)

    def test_resize_boxes(self):
        boxes = image.resize_boxes([0, 0, 0, 0, 1], xy=(0, 0), scale=1)
        self.assertEqual([0, 0, 0, 0, 1], boxes)

        xy = (52, 0)
        scale = 0.541666687
        box = [15.5, 224.5, 552.5, 429.5, 2]

        boxes = image.resize_boxes(box, xy=xy, scale=scale)
        x_min, y_min, x_max, y_max, classes = boxes.numpy()

        self.assertAlmostEqual(float("{0:.3f}".format((box[0] * scale) + xy[0])), x_min, places=3)
        self.assertAlmostEqual(float("{0:.3f}".format((box[1] * scale) + xy[1])), y_min, places=3)
        self.assertAlmostEqual(float("{0:.3f}".format((box[2] * scale) + xy[0])), x_max, places=3)
        self.assertAlmostEqual(float("{0:.3f}".format((box[3] * scale) + xy[1])), y_max, places=3)
        self.assertAlmostEqual(2.0, classes, places=1)

    def test_resize_image_and_boxes(self):
        content = image.to_bytes(self._template)
        serialize = self._feature.serialize(content, b'label', size=(self._template.height, self._template.width))
        images, labels, extras = self._feature.deserialize(serialize)

        size = (416, 416)
        box = [[15.5, 224.5, 552.5, 429.5, 2]]
        scale = min(size[0] / self._template.height, size[1] / self._template.width)

        new_height = scale * self._template.height
        new_width = scale * self._template.width

        xy = (size[1] - new_width) / 2, (size[0] - new_height) / 2

        images, boxes = image.resize_image_and_boxes(images, size=size, boxes=box)
        boxes = tf.reshape(boxes, shape=[-1])
        x_min, y_min, x_max, y_max, classes = boxes.numpy()

        self.assertEqual((size[0], size[1], 3), images.shape)
        self.assertAlmostEqual(float("{0:.3f}".format((box[0][0] * scale) + xy[0])), x_min, places=3)
        self.assertAlmostEqual(float("{0:.3f}".format((box[0][1] * scale) + xy[1])), y_min, places=3)
        self.assertAlmostEqual(float("{0:.3f}".format((box[0][2] * scale) + xy[0])), x_max, places=3)
        self.assertAlmostEqual(float("{0:.3f}".format((box[0][3] * scale) + xy[1])), y_max, places=3)
        self.assertAlmostEqual(2.0, classes, places=1)

    def test_from_array(self):
        content = image.to_bytes(self._template)
        serialize = self._feature.serialize(content, b'label', size=(self._template.height, self._template.width))
        images, labels, extras = self._feature.deserialize(serialize)

        img = image.from_array(images.numpy())
        self.assertEqual(576, img.width)
        self.assertEqual(768, img.height)

    def test_draw(self):
        content = image.to_bytes(self._template)
        serialize = self._feature.serialize(content, b'label', size=(self._template.height, self._template.width))
        images, labels, extras = self._feature.deserialize(serialize)

        boxes = [[[10, 30, 200, 100, 1], [20, 120, 400, 200, 2]]]
        images_draw = image.draw(images, boxes, classes=path.get_class_name())

        self.assertFalse(np.array_equal(images.numpy(), images_draw[0].numpy()))
        self.assertEqual(576, images_draw.shape[2])
        self.assertEqual(768, images_draw.shape[1])

    def test_draw_text_center(self):
        font = path.get_fonts()[0]
        text = path.load_txt(path.resolve('tests/resources/1.txt'))

        layer = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        image.draw_text(layer, text, font, align='center')

        boxes = image.generate_boxes(layer, text, font, align='center')
        self.assertEqual(8, len(boxes))
        self.assertEqual([142.5, 48, 433.5, 84, 0], boxes[0])
        self.assertEqual([243.5, 96, 332.5, 132, 0], boxes[1])

    def test_draw_text(self):
        font = path.get_fonts()[0]
        text = path.load_txt(path.resolve('tests/resources/1.txt'))

        layer = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        image.draw_text(layer, text, font)

        boxes = image.generate_boxes(layer, text, font)
        self.assertEqual(8, len(boxes))
        self.assertEqual([48, 48, 339, 84, 0], boxes[0])
        self.assertEqual([48, 96, 137, 132, 0], boxes[1])
        self.assertEqual([48, 272, 414, 308, 0], boxes[4])

    def test_get_size_boxes(self):
        size = image.get_size_boxes([[5, 5, 20, 50]])
        self.assertTrue(np.array_equal([[[15, 45]]], size.numpy()))

        size = image.get_size_boxes([10, 40, 60, 120])
        self.assertTrue(np.array_equal([50, 80], size.numpy()))

    def test_get_max_size_boxes(self):
        size = image.get_max_size_boxes([[5, 5, 20, 50]])
        self.assertTrue(np.array_equal([15, 45], size.numpy()))

        size = image.get_max_size_boxes([[10, 40, 60, 120], [5, 5, 20, 50]])
        self.assertTrue(np.array_equal([50, 80], size.numpy()))

    def test_crop_and_resize(self):
        font = path.get_fonts()[0]
        text = path.load_txt(path.resolve('tests/resources/1.txt'))

        layer = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        image.draw_text(layer, text, font)

        boxes = image.generate_boxes(layer, text, font)
        layer = image.to_decode(image.to_bytes(layer), shape=(768, 576, 3))

        boxes = image.crop_and_resize(layer, boxes)
        self.assertEqual([8, 180, 366, 3], boxes[0][1].get_shape())

    def test_resize_with_crop_or_pad(self):
        layer = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        layer = image.to_decode(image.to_bytes(layer), shape=(768, 576, 3))

        layer = image.crop(layer, (5, 10, 25, 50), (50, 50))
        self.assertEqual([50, 50, 3], layer.get_shape())

    def test_boxes_characters(self):
        font = path.get_fonts()[0]
        text = path.load_txt(path.resolve('tests/resources/3.txt'))

        layer = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        image.draw_text(layer, text, font)

        boxes = image.generate_boxes(layer, text, font)
        self.assertTrue(np.array_equal(
            [[48, 48, 231, 84, 0], [48, 96, 137, 132, 0], [60, 192, 373, 436, 0], [48, 448, 350, 484, 0]], boxes))

    def test_boxes_size(self):
        font = path.get_fonts()[0]
        text = path.load_txt(path.resolve('tests/resources/2.txt'))

        layer = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        image.draw_text(layer, text, font)

        boxes = image.generate_boxes(layer, text, font)

        layer = image.to_decode(image.to_bytes(layer), shape=(768, 576, 3))
        layer, boxes = image.resize_image_and_boxes(layer, size=(416, 416), boxes=boxes)

        boxes = tf.reshape(boxes, shape=[-1])
        self.assertTrue(np.allclose(
            [81.25, 26., 243.20834, 158.16667, 0., 81.25, 173.33334, 265.4167, 305.5, 0.], boxes.numpy()))


if __name__ == '__main__':
    unittest.main()
