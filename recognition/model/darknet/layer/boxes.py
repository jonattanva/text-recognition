# coding: utf-8
import tensorflow as tf


class Boxes(tf.keras.layers.Layer):
    NUMBER_CLASSES = 1

    def __init__(self, anchors, anchors_masks, **kwargs):
        super(Boxes, self).__init__(name=kwargs.get('name', 'boxes'))
        self._anchors = anchors
        self._anchors_masks = anchors_masks
        self._number_classes = kwargs.get('number_classes', Boxes.NUMBER_CLASSES)

        self._boxes1 = tf.keras.layers.Lambda(
            lambda value: self.convert_by_index_anchor(value, 0))

        self._boxes2 = tf.keras.layers.Lambda(
            lambda value: self.convert_by_index_anchor(value, 1))

        self._boxes3 = tf.keras.layers.Lambda(
            lambda value: self.convert_by_index_anchor(value, 2))

    def build(self, input_shape):
        super(Boxes, self).build(input_shape)

    def get_config(self):
        config = super(Boxes, self).get_config()
        config.update({'anchors': self._anchors})
        config.update({'anchors_masks': self._anchors_masks})
        config.update({'number_classes': self._number_classes})
        return config

    def convert_by_index_anchor(self, inputs, index):
        return self.convert(inputs, self._anchors[self._anchors_masks[index]])

    def convert(self, inputs, anchors):
        # inputs: [batch_size, grid_size, grid_size, anchors, [x, y, w, h, obj, ...classes]]
        box_xy, box_wh, objectness, class_score = tf.split(
            inputs, num_or_size_splits=(2, 2, 1, self._number_classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_score = tf.sigmoid(class_score)
        prediction_box = tf.concat([box_xy, box_wh], axis=-1)

        grid_size = tf.shape(inputs)[1]
        grid_range = tf.range(grid_size)
        grid = tf.meshgrid(grid_range, grid_range)
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

        box_xy = tf.math.add(box_xy, tf.cast(grid, dtype=tf.float32))
        box_xy = tf.math.divide(box_xy, tf.cast(grid_size, dtype=tf.float32))
        box_wh = tf.math.multiply(tf.math.exp(box_wh), anchors)

        box_x1y1 = tf.math.divide(tf.math.subtract(box_xy, box_wh), 2)
        box_x2y2 = tf.math.divide(tf.math.add(box_xy, box_wh), 2)

        box = tf.concat([box_x1y1, box_x2y2], axis=-1)
        return box, objectness, class_score, prediction_box

    def call(self, inputs, training=False):
        output_1, output_2, output_3 = inputs
        if not training:
            output_1 = self._boxes1(output_1)
            output_2 = self._boxes2(output_2)
            output_3 = self._boxes3(output_3)
        return output_1, output_2, output_3
