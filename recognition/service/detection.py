# coding: utf-8
import tensorflow as tf
import recognition.util.image
import recognition.util.math as math

from recognition.service.training import Training
from recognition.model.darknet.yolov3 import YoloV3


class Detection(Training):
    """Se encarga de detectar y clasificar el texto de una imagen"""

    def __init__(self, anchors, classes, **kwargs):
        self._classes = classes
        self._anchors = anchors
        self._number_classes = len(classes)
        super(Detection, self).__init__(model=YoloV3(classes=classes, anchors=anchors),
                                        strategy=kwargs.get('strategy', None),
                                        destination_path=kwargs.get('destination_path', None))

    def loss_step(self, labels, predictions):
        prediction_loss = []
        regularization_loss = tf.math.reduce_sum(self._model.losses)
        for label, prediction, mask in zip(labels, predictions, YoloV3.ANCHOR_MASKS):
            prediction_loss.append(self.calculate_loss(label, prediction, self._anchors[mask]))
        total_loss = tf.math.add(tf.math.reduce_sum(prediction_loss), regularization_loss)
        return total_loss

    def accuracy_step(self, labels, predictions):
        prediction_accuracy = []
        for label, prediction, mask in zip(labels, predictions, YoloV3.ANCHOR_MASKS):
            prediction_accuracy.append(self.calculate_accuracy(label, prediction, self._anchors[mask]))
        total_accuracy = tf.math.reduce_sum(prediction_accuracy)
        return total_accuracy

    def calculate_accuracy(self, label, prediction, anchors):
        """Calcula la precisión del modelo"""

        # labels: [batch_size, grid_size, grid_size, anchors, [x_min, y_min, x_max, y_max, obj, class]]
        true_box, true_obj, true_class_idx = Detection.prepare_label(label)

        # prediction: [batch_size, grid, grid, anchors, [x, y, w, h, obj, ...class]]
        bbox, objectness, class_score, prediction_box = self._model.convert(inputs=prediction, anchors=anchors)

        class_accuracy = tf.keras.metrics.categorical_accuracy(true_class_idx, prediction_box)
        class_accuracy = tf.math.reduce_sum(class_accuracy, axis=(1, 2, 3))

        return class_accuracy

    def calculate_loss(self, label, prediction, anchors):
        """Calcula la pérdida del modelo"""

        # labels: [batch_size, grid_size, grid_size, anchors, [x_min, y_min, x_max, y_max, obj, class]]
        true_box, true_obj, true_class_idx = Detection.prepare_label(label)
        true_xy = tf.math.divide(tf.math.add(true_box[..., 0:2], true_box[..., 2:4]), 2)
        true_wh = recognition.util.image.get_size_boxes(true_box)

        # prediction: [batch_size, grid, grid, anchors, [x, y, w, h, obj, ...class]]
        bbox, objectness, class_score, prediction_box = self._model.convert(inputs=prediction, anchors=anchors)
        prediction_xy = prediction_box[..., 0:2]
        prediction_wh = prediction_box[..., 2:4]

        # Ignora los valores que superan el umbral
        obj_mask = tf.squeeze(true_obj, axis=-1)
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, dtype=tf.bool))
        best_iou = tf.math.reduce_max(math.broadcast_iou(bbox, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < YoloV3.IGNORE_THRESH, dtype=tf.float32)

        # Obtiene la pérdida
        xy_loss = obj_mask * tf.math.reduce_sum(tf.math.square(true_xy - prediction_xy), axis=-1)
        wh_loss = obj_mask * tf.math.reduce_sum(tf.math.square(true_wh - prediction_wh), axis=-1)

        obj_loss = tf.keras.losses.binary_crossentropy(true_obj, objectness)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * tf.keras.losses.binary_crossentropy(true_class_idx, prediction_box)

        xy_loss = tf.math.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.math.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.math.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.math.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    @staticmethod
    def prepare_label(value):
        return tf.split(tf.squeeze(value), num_or_size_splits=(4, 1, 1), axis=-1)
