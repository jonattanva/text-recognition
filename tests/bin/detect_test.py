import unittest
import PIL.Image
import recognition.util.image
import recognition.util.path as path

from recognition.bin.detect import Detect
from recognition.model.darknet.yolov3 import YoloV3


class GenerateTestCase(unittest.TestCase):

    def setUp(self):
        self.classes = path.get_class_name()
        self.anchor_masks = path.get_anchors()

    def test_detect(self):
        original = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        original = recognition.util.image.to_bytes(original)
        original = recognition.util.image.to_decode(original, shape=(768, 576, 3))

        filename = path.resolve('temp/test/example.jpg')
        recognition.util.image.save(original.numpy() * 255, filename=filename)

        yolo = YoloV3(classes=self.classes, anchors=self.anchor_masks)
        yolo.build(input_shape=(YoloV3.IMAGE_SIZE, YoloV3.IMAGE_SIZE, 3))

        filepath = path.resolve('temp/test/model/{}.h5'.format(yolo.name))
        images = recognition.util.image.resize(original, (YoloV3.IMAGE_SIZE, YoloV3.IMAGE_SIZE))

        yolo(images, training=True)
        yolo.save_weights(filepath)

        destination_path = path.resolve('temp/test/output.jpg')
        detect = Detect(filename=filename, destination_path=destination_path, model_path=filepath)
        detect.start()

        self.assertTrue(filename)
        self.assertTrue(path.exists(destination_path))


if __name__ == '__main__':
    unittest.main()
