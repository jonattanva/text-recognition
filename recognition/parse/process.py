# coding: utf-8
import tensorflow as tf
import recognition.util.image

from recognition.parse.feature import Feature
from recognition.model.darknet.yolov3 import YoloV3


class Process(Feature):
    """Convierte el set de datos que se utiliza en el modelo Yolo v3"""

    GRID_SIZE = 13

    def __init__(self, anchors, channel=None):
        super(Process, self).__init__(channel=channel)
        self._anchors = anchors / YoloV3.IMAGE_SIZE

    def deserialize(self, serialized):
        # images: [height, width, channel]
        images, label, extras = super(Process, self).deserialize(serialized)
        images, boxes = recognition.util.image.resize_image_and_boxes(
            images, size=(YoloV3.IMAGE_SIZE, YoloV3.IMAGE_SIZE), boxes=extras[Feature.LOCATION_BBOX])

        max_pad = extras[Feature.LOCATION_BBOX_MAX_LENGTH]
        max_pad = tf.math.subtract(max_pad, tf.shape(boxes)[1])

        # boxes: [batch_size, boxes, [x_min, y_min, x_max, y_max, class]]
        boxes = tf.pad(boxes, [[0, 0], [0, max_pad], [0, 0]])

        # boxes: [batch_size, boxes, [x_min, y_min, x_max, y_max, class, anchor_id]]
        anchor_index = self.get_anchor_index(boxes)
        boxes = tf.concat([boxes, anchor_index], axis=-1)

        result = []
        grid_size = Process.GRID_SIZE
        for mask in YoloV3.ANCHOR_MASKS:
            result.append(Process.convert(boxes, grid_size, mask))
            grid_size = grid_size * 2

        return images, tuple(result)

    def padded_shapes(self):
        return None

    def get_anchor_index(self, boxes):
        """Obtiene el id de la ancla que corresponda a la IOU"""
        box_wh = recognition.util.image.get_size_boxes(boxes)
        box_wh = tf.tile(tf.expand_dims(box_wh, axis=-2), (1, 1, tf.shape(self._anchors)[0], 1))

        box_area = tf.math.multiply(box_wh[..., 0], box_wh[..., 1])

        anchor_area = tf.math.multiply(self._anchors[..., 0], self._anchors[..., 1])
        intersection = tf.math.multiply(tf.minimum(box_wh[..., 0], self._anchors[..., 0]),
                                        tf.minimum(box_wh[..., 1], self._anchors[..., 1]))

        iou = intersection / (box_area + anchor_area - intersection)

        anchor_id = tf.cast(tf.argmax(iou, axis=-1), dtype=tf.float32)
        anchor_id = tf.expand_dims(anchor_id, axis=-1)
        return anchor_id

    @staticmethod
    def convert(boxes, grid_size, mask):
        """Convierte los cuadros delimitadores en un tensor con la misma forma de la predicción"""
        mask = tf.cast(mask, dtype=tf.int32)
        if not tf.is_tensor(boxes):
            boxes = tf.convert_to_tensor(boxes)

        # boxes: [batch_size, boxes, [x_min, y_min, x_max, y_max, class, anchor_id]]
        number_boxes = tf.shape(boxes)[0]

        # template: [number_boxes, grid_size, grid_size, anchors, [x, y, w, h, obj, class]]
        template = tf.zeros([number_boxes, grid_size, grid_size, tf.shape(mask)[0], 6])

        @tf.function
        def runnable():
            step = 0
            indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
            updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

            for i in tf.range(number_boxes):
                for j in tf.range(tf.shape(boxes)[1]):
                    if tf.math.equal(tf.math.reduce_sum(boxes[i][j]), 0):
                        continue

                    anchor_eq = tf.math.equal(mask, tf.cast(boxes[i][j][5], dtype=tf.int32))
                    if tf.math.reduce_any(anchor_eq):
                        box = boxes[i][j][0:4] / YoloV3.IMAGE_SIZE

                        box_xy = tf.math.add(box[0:2], box[2:4])
                        box_xy = tf.math.divide(box_xy, 2)

                        anchor_id = tf.cast(tf.where(anchor_eq), dtype=tf.int32)
                        grid_xy = tf.cast(tf.math.floordiv(box_xy, tf.math.divide(1, grid_size)), dtype=tf.int32)

                        indexes = indexes.write(step, [i, grid_xy[1], grid_xy[0], anchor_id[0][0]])
                        updates = updates.write(step, [box[0], box[1], box[2], box[3], 1, boxes[i][j][4]])
                        step = step + 1

            return tf.tensor_scatter_nd_update(template, indexes.stack(), updates.stack())

        return runnable()
