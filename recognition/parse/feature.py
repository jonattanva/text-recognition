# coding: utf-8
import uuid
import numpy as np
import tensorflow as tf
import recognition.util.image


class Feature:
    """Serializar y deserializar un archivo de tipo tfrecords"""

    CHANNEL = 3
    VERSION_NUMBER = 1

    TOKEN = "token"
    IMAGE = "image"
    IMAGE_WIDTH = "image/width"
    IMAGE_HEIGHT = "image/height"
    CLASSIFICATION_LABEL = "classification/label"
    CLASSIFICATION_LABEL_MAX_LENGTH = "classification/label/length"
    CLASSIFICATION_LABEL_ALIGNMENT = "classification/label/alignment"
    LOCATION_BBOX = "location/bbox"
    LOCATION_BBOX_MAX_LENGTH = "location/bbox/length"
    VERSION = "version"

    ALIGNMENT_LEFT = "left"
    ALIGNMENT_RIGHT = "right"
    ALIGNMENT_CENTER = "center"

    def __init__(self, channel=None, encoding='utf-8'):
        if channel is None:
            channel = Feature.CHANNEL

        self._channel = channel
        self._encoding = encoding
        self._version = Feature.VERSION_NUMBER

    def serialize(self, image, label, size, **kwargs):
        """Crea un mensaje de tipo tf.train.Example listo para ser escrito en un archivo .tfrecords"""
        height, width = size

        if not isinstance(image, bytes):
            raise ValueError('The image is not byte type')

        token = kwargs.get("token", uuid.uuid1())
        classification_label_alignment = kwargs.get("alignment", Feature.ALIGNMENT_LEFT)
        location_bbox = kwargs.get("bbox", [])
        location_bbox_max_length = kwargs.get("bbox_max_length", 0)

        feature = {
            Feature.TOKEN: self.bytes_feature(token),
            Feature.IMAGE: self.bytes_feature(image),
            Feature.IMAGE_HEIGHT: Feature.int64_feature(height),
            Feature.IMAGE_WIDTH: Feature.int64_feature(width),
            Feature.CLASSIFICATION_LABEL: self.bytes_feature(label),
            Feature.CLASSIFICATION_LABEL_MAX_LENGTH: Feature.int64_feature(len(label)),
            Feature.CLASSIFICATION_LABEL_ALIGNMENT: self.bytes_feature(classification_label_alignment),
            Feature.LOCATION_BBOX: Feature.float_feature(location_bbox),
            Feature.LOCATION_BBOX_MAX_LENGTH: Feature.int64_feature(location_bbox_max_length),
            Feature.VERSION: Feature.int64_feature(self._version)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def deserialize(self, serialized):
        """Obtiene las propiedades escritas en un archivo .tfrecords"""
        feature = {
            Feature.TOKEN: tf.io.FixedLenFeature([], tf.string, default_value=''),
            Feature.IMAGE: tf.io.FixedLenFeature([], tf.string, default_value=''),
            Feature.IMAGE_HEIGHT: tf.io.FixedLenFeature([], tf.int64, default_value=0),
            Feature.IMAGE_WIDTH: tf.io.FixedLenFeature([], tf.int64, default_value=0),
            Feature.CLASSIFICATION_LABEL: tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
            Feature.CLASSIFICATION_LABEL_MAX_LENGTH: tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
            Feature.CLASSIFICATION_LABEL_ALIGNMENT: tf.io.FixedLenFeature(
                [], dtype=tf.string, default_value=Feature.ALIGNMENT_LEFT),
            Feature.LOCATION_BBOX: tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            Feature.LOCATION_BBOX_MAX_LENGTH: tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
            Feature.VERSION: tf.io.FixedLenFeature([], dtype=tf.int64, default_value=self._version)
        }

        example = tf.io.parse_single_example(serialized, features=feature)

        width = example[Feature.IMAGE_WIDTH]
        width = tf.cast(width, dtype=tf.int16)

        height = example[Feature.IMAGE_HEIGHT]
        height = tf.cast(height, dtype=tf.int16)

        image = example[Feature.IMAGE]  # [height, width, channel]
        image = recognition.util.image.to_decode(image, shape=(height, width, self._channel))

        token = example[Feature.TOKEN]
        label = example[Feature.CLASSIFICATION_LABEL]
        alignment = example[Feature.CLASSIFICATION_LABEL_ALIGNMENT]

        version = example[Feature.VERSION]
        version = tf.cast(version, dtype=tf.int8)

        label_max_length = example[Feature.CLASSIFICATION_LABEL_MAX_LENGTH]
        label_max_length = tf.cast(label_max_length, dtype=tf.int32)

        location_bbox_max_length = example[Feature.LOCATION_BBOX_MAX_LENGTH]
        location_bbox_max_length = tf.cast(location_bbox_max_length, dtype=tf.int32)

        location_bbox = example[Feature.LOCATION_BBOX]  # [x_min, y_min, x_max, y_max, class]
        location_bbox = tf.reshape(location_bbox, shape=[-1, 5])

        extras = dict()
        extras[Feature.TOKEN] = token
        extras[Feature.IMAGE_HEIGHT] = height
        extras[Feature.IMAGE_WIDTH] = width
        extras[Feature.LOCATION_BBOX] = location_bbox
        extras[Feature.CLASSIFICATION_LABEL_MAX_LENGTH] = label_max_length
        extras[Feature.LOCATION_BBOX_MAX_LENGTH] = location_bbox_max_length
        extras[Feature.CLASSIFICATION_LABEL_ALIGNMENT] = alignment
        extras[Feature.VERSION] = version

        return image, label, extras

    def padded_shapes(self):
        """Representa la forma en la que se debe de llenar el set de datos"""
        return ([None, None, None], [], {
            Feature.LOCATION_BBOX: [None, None],
            Feature.LOCATION_BBOX_MAX_LENGTH: [],
            Feature.CLASSIFICATION_LABEL_MAX_LENGTH: [],
            Feature.TOKEN: [],
            Feature.IMAGE_HEIGHT: [],
            Feature.IMAGE_WIDTH: [],
            Feature.VERSION: [],
            Feature.CLASSIFICATION_LABEL_ALIGNMENT: []
        })

    def bytes_feature(self, value):
        """Devuelve una lista de bytes desde un string - UUID - byte"""
        if isinstance(value, uuid.UUID):
            value = bytes(str(value), encoding=self._encoding)

        if isinstance(value, str):
            value = bytes(value, encoding=self._encoding)

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def int64_feature(value):
        """Devuelve una lista de enteros desde un bool - enum - int - uint"""
        if isinstance(value, list):
            value = np.array(value).ravel()

        if isinstance(value, int):
            value = [value]

        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        """Devuelve una lista de flotantes desde un float - double."""
        if isinstance(value, list):
            value = np.array(value).ravel()

        if isinstance(value, float):
            value = [value]

        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
