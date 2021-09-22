# coding: utf-8
import time
import argparse
import numpy as np
import tensorflow as tf
import recognition.util.image
import recognition.util.path as path
import recognition.util.utilities as utilities

from recognition.model.darknet.yolov3 import YoloV3


class Detect:
    PATH_OUTPUT = 'temp/output.jpg'

    def __init__(self, filename, destination_path=None, model_path=None, debug=False):
        if destination_path is None:
            destination_path = path.resolve(Detect.PATH_OUTPUT)

        self._debug = debug
        self._filename = filename
        self._model_path = model_path
        self._destination_path = destination_path
        self._classes = path.get_class_name()
        self._anchor_masks = path.get_anchors()

    def start(self):
        if self._debug:
            tf.debugging.set_log_device_placement(True)

        start = time.time()
        image = recognition.util.image.from_file(self._filename)
        width, height = image.size

        image = recognition.util.image.to_bytes(image)
        image = recognition.util.image.to_decode(image, shape=[height, width, 3])

        model = YoloV3(classes=self._classes, anchors=self._anchor_masks)
        model.build(input_shape=(YoloV3.IMAGE_SIZE, YoloV3.IMAGE_SIZE, 3))
        model.load_weights(self._model_path)

        inputs = recognition.util.image.resize(image, size=(YoloV3.IMAGE_SIZE, YoloV3.IMAGE_SIZE))
        output_1, output_2, output_3 = model(inputs, training=False)

        boxes, scores, classes, prediction_box = YoloV3.combined_bounding_boxes(
            outputs=(output_1[:3], output_2[:3], output_3[:3]))

        bbox = []
        for box in range(prediction_box[0]):
            key_class = classes[0][box]
            x_min, y_min, x_max, y_max = boxes[0][box]
            bbox.append(tf.stack([x_min, y_min, x_max, y_max, key_class], axis=-1))

            key_value = utilities.get_key_from_dictionary(key_class, self._classes)
            print('\t{}, {}, {}'.format(key_value, np.array(scores[0][box]), np.array(boxes[0][box])))

        if len(bbox) == 0:
            print('It has no bounding boxes')

        image = recognition.util.image.draw(image, bbox, classes=self._classes, normalize=True)
        recognition.util.image.save(image.numpy() * 255, self._destination_path)

        print('Time {} sec'.format(time.time() - start))
        print('Output saved to: {}'.format(self._destination_path))


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        metavar='',
                        help='set if device placements should be logged')

    parser.add_argument('--filename',
                        type=str,
                        default=None,
                        metavar='',
                        help='')

    parser.add_argument('--destination-path',
                        type=str,
                        metavar='',
                        default=None,
                        help='')

    parser.add_argument('--model-path',
                        type=str,
                        metavar='',
                        default=None,
                        help='')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = process_args()

    detect = Detect(arguments.filename,
                    debug=arguments.debug,
                    destination_path=arguments.destination_path,
                    model_path=arguments.model_path)

    detect.start()
