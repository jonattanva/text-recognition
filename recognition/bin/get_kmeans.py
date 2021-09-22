# coding: utf-8
import argparse
import tensorflow as tf
import recognition.util.kmeans as kmeans

from recognition.parse.feature import Feature
from recognition.service.dataset import Dataset

CLUSTER_NUMBER = 9


def get_kmeans(boxes, cluster_number=None):
    """Obtiene los k-means de las cajas delimitadoras"""
    if cluster_number is None:
        cluster_number = CLUSTER_NUMBER

    anchors = kmeans.kmeans(boxes, cluster_number)
    ave_iou = kmeans.avg_iou(boxes, anchors)

    anchors = anchors.astype('int').tolist()
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])
    return anchors, ave_iou


def get_boxes(filename, compression_type=None):
    """Obtiene las cajas delimitadoras del set de datos"""
    if compression_type is None:
        compression_type = Dataset.COMPRESSION_TYPE

    feature = Feature(channel=Feature.CHANNEL)
    dataset = Dataset(filename, feature, compression_type=compression_type)
    dataset = dataset.load()
    dataset = dataset.map(feature.deserialize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    results = kmeans.clustering_data(dataset)
    return results


def anchors_prepare(anchors):
    anchor_value = ''
    for anchor in anchors:
        anchor_value += '{},{}, '.format(anchor[0], anchor[1])
    anchor_value = anchor_value[:-2]
    return anchor_value


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename',
                        type=str,
                        default=None,
                        metavar='',
                        help='')

    parser.add_argument('--compression-type',
                        type=str,
                        metavar='',
                        default='GZIP',
                        help='type compression (ZLIB or GZIP)')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = process_args()

    boxes_result = get_boxes(
        filename=arguments.filename, compression_type=arguments.compression_type)

    anchors_result, ave_iou_result = get_kmeans(
        boxes_result, cluster_number=CLUSTER_NUMBER)
    anchor_string = anchors_prepare(anchors_result)

    print('Anchors are: {}'.format(anchor_string))
    print('The average iou is: {}'.format(ave_iou_result))
