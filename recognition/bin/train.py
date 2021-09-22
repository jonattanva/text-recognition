# coding: utf-8
import argparse
import tensorflow as tf
import recognition.util.path as path

from recognition.service.dataset import Dataset
from recognition.parse.process import Process
from recognition.service.detection import Detection


class Train:

    def __init__(self, input_shape, **kwargs):
        self._debug = kwargs.get('debug', False)
        self._memory_growth = kwargs.get('memory_growth', False)

        tf.debugging.set_log_device_placement(self._debug)
        if self._memory_growth:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        self._input_shape = input_shape  # (height, width, channel)
        self._buffer_size = kwargs.get('buffer_size', 1024)
        self._epoch = kwargs.get('epoch', 1)
        self._destination_path = kwargs.get('destination_path', None)
        self._compression_type = kwargs.get('compression_type', Dataset.COMPRESSION_TYPE)
        self._max_outputs = kwargs.get('max_outputs', Detection.MAX_OUTPUTS)
        self._filename = kwargs.get('filename', Dataset.load_filename())
        self._strategy = tf.distribute.MirroredStrategy()
        self._batch_size = kwargs.get('batch_size', 64) * self._strategy.num_replicas_in_sync

    def start(self):
        anchors = path.get_anchors()
        classes = path.get_class_name()

        height, width, channel = self._input_shape
        transform = Process(channel=channel, anchors=anchors)

        with self._strategy.scope():
            dataset = Dataset(feature=transform,
                              filename=self._filename,
                              compression_type=self._compression_type)

            dataset = dataset(epoch=self._epoch,
                              batch_size=self._batch_size,
                              buffer_size=self._buffer_size)

            detection = Detection(anchors=anchors,
                                  classes=classes,
                                  strategy=self._strategy,
                                  destination_path=self._destination_path)

            detection.checkpoint_manager()

            dataset = self._strategy.experimental_distribute_dataset(dataset)
            detection.start(dataset, max_outputs=self._max_outputs)


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

    parser.add_argument('--width',
                        type=int,
                        default=576,
                        metavar='',
                        help="width of the image")

    parser.add_argument('--height',
                        type=int,
                        default=768,
                        metavar='',
                        help="height of the image")

    parser.add_argument('--channel',
                        type=int,
                        default=3,
                        metavar='',
                        help="image channels")

    parser.add_argument('--buffer-size',
                        type=int,
                        metavar='',
                        default=1024,
                        help='')

    parser.add_argument('--batch-size',
                        type=int,
                        metavar='',
                        default=64,
                        help='')

    parser.add_argument('--epoch',
                        type=int,
                        metavar='',
                        default=1,
                        help='')

    parser.add_argument('--compression-type',
                        type=str,
                        metavar='',
                        default='GZIP',
                        help='type compression (ZLIB or GZIP)')

    parser.add_argument('--max_outputs',
                        type=int,
                        metavar='',
                        default=3,
                        help='')

    parser.add_argument('--destination-path',
                        type=str,
                        metavar='',
                        default=None,
                        help='')

    parser.add_argument('--memory-growth',
                        type=bool,
                        default=False,
                        metavar='',
                        help='')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = process_args()
    train = Train(
        input_shape=(arguments.height, arguments.width, arguments.channel),
        filename=arguments.filename,
        buffer_size=arguments.buffer_size,
        batch_size=arguments.batch_size,
        epoch=arguments.epoch,
        compression_type=arguments.compression_type,
        max_outputs=arguments.max_outputs,
        destination_path=arguments.destination_path,
        debug=arguments.debug,
        memory_growth=arguments.memory_growth)

    train.start()
