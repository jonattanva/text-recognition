# coding: utf-8
import tensorflow as tf


def broadcast_iou(box_1, box_2):
    """Obtiene "Intersection over Union" (IoU) entre dos cajas delimitadoras"""
    box_1 = tf.expand_dims(box_1, axis=-2)
    box_2 = tf.expand_dims(box_2, axis=0)

    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, shape=new_shape)
    box_2 = tf.broadcast_to(box_2, shape=new_shape)

    int_w = tf.math.maximum(tf.math.minimum(box_1[..., 2], box_2[..., 2]) -
                            tf.math.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.math.maximum(tf.math.minimum(box_1[..., 3], box_2[..., 3]) -
                            tf.math.maximum(box_1[..., 1], box_2[..., 1]), 0)

    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

    return int_area / (box_1_area + box_2_area - int_area)
