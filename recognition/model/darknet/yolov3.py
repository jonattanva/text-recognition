# coding: utf-8
import numpy as np
import tensorflow as tf

from recognition.model.model import Model
from recognition.model.darknet.layer.boxes import Boxes
from recognition.model.darknet.layer.wrapper import Wrapper
from recognition.model.darknet.layer.prepare import Prepare
from recognition.model.darknet.backbone.darknet53 import Darknet53


class YoloV3(Model):
    IMAGE_SIZE = 416
    NUMBER_ANCHORS = 3
    IGNORE_THRESH = 0.5
    ANCHOR_MASKS = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    def __init__(self, classes, anchors, number_anchors=None):
        super(YoloV3, self).__init__(name='yolo_v3')
        if number_anchors is None:
            number_anchors = YoloV3.NUMBER_ANCHORS

        self._classes = classes
        self._anchors = anchors
        self._number_classes = len(classes)
        self._number_anchors = number_anchors

        self._darknet53 = Darknet53()
        self._conv1 = Wrapper(filters=512, name='conv_1')
        self._conv2 = Wrapper(filters=256, name='conv_2')
        self._conv3 = Wrapper(filters=128, name='conv_3')

        self._prepare1 = Prepare(
            filters=512,
            number_anchors=self._number_anchors,
            number_classes=self._number_classes,
            name='output_1')

        self._prepare2 = Prepare(
            filters=256,
            number_anchors=self._number_anchors,
            number_classes=self._number_classes,
            name="output_2")

        self._prepare3 = Prepare(
            filters=128,
            number_anchors=self._number_anchors,
            number_classes=self._number_classes,
            name='output_3')

        self._boxes = Boxes(
            anchors=self._anchors,
            anchors_masks=YoloV3.ANCHOR_MASKS,
            number_classes=self._number_classes,
            name='boxes')

    def build(self, input_shape):
        super(YoloV3, self).build(input_shape)

    def convert(self, inputs, anchors):
        return self._boxes.convert(inputs, anchors)

    @staticmethod
    def reshape(output, index):
        output = output[index]
        output_shape = tf.shape(output)
        return tf.reshape(output, shape=(output_shape[0], -1, output_shape[-1]))

    @staticmethod
    def combined_bounding_boxes(outputs):
        boxes, confidence, prediction_box = [], [], []
        for output in outputs:
            boxes.append(YoloV3.reshape(output, index=0))
            confidence.append(YoloV3.reshape(output, index=1))
            prediction_box.append(YoloV3.reshape(output, index=2))

        confidence = tf.concat(confidence, axis=1)
        prediction_box = tf.concat(prediction_box, axis=1)

        scores = confidence * prediction_box
        scores_shape = tf.shape(scores)
        scores = tf.reshape(scores, shape=(scores_shape[0], -1, scores_shape[-1]))

        boxes = tf.concat(boxes, axis=1)
        boxes_shape = tf.shape(boxes)
        boxes = tf.reshape(boxes, shape=(boxes_shape[0], -1, 1, 4))

        # [y1, x1, y2, x2]
        boxes, scores, classes, prediction_box = tf.image.combined_non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_total_size=100,
            max_output_size_per_class=100,
            iou_threshold=YoloV3.IGNORE_THRESH,
            score_threshold=YoloV3.IGNORE_THRESH)

        return boxes, scores, classes, prediction_box

    def call(self, inputs, training=False, mask=None):
        inputs_shape = inputs.get_shape()
        if inputs_shape.ndims == 3:
            inputs = tf.expand_dims(inputs, axis=0)

        inputs_a, inputs_b, inputs = self._darknet53(inputs, training=training)

        inputs = self._conv1(inputs, training=training)
        output_1 = self._prepare1(inputs, training=training)

        inputs = self._conv2((inputs, inputs_b), training=training)
        output_2 = self._prepare2(inputs, training=training)

        inputs = self._conv3((inputs, inputs_a), training=training)
        output_3 = self._prepare3(inputs, training=training)

        output_1, output_2, output_3 = self._boxes(inputs=(output_1, output_2, output_3), training=training)
        return output_1, output_2, output_3
