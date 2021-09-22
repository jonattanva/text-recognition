# coding: utf-8
import tensorflow as tf
import recognition.util.path as path

from recognition.parse.feature import Feature


class Dataset:
    """Se encarga de cargar y procesar el set de datos"""

    CYCLE_LENGTH = 2
    VERSION_NUMBER = "v1"
    COMPRESSION_TYPE = 'GZIP'
    PATH_CACHE = "temp/dataset"

    def __init__(self, filename, feature, compression_type=None):
        if not isinstance(feature, Feature):
            raise ValueError('the feature are not an instance of recognition.service.feature')

        if compression_type is None:
            compression_type = Dataset.COMPRESSION_TYPE

        self._feature = feature
        self._filename = filename
        self._compression_type = compression_type

    def record_dataset(self, filename, num_parallel_reads=None):
        """Un conjunto de datos que comprende registros de uno o más archivos TFRecord."""
        return tf.data.TFRecordDataset(filename, compression_type=self._compression_type,
                                       num_parallel_reads=num_parallel_reads)

    def load(self, cache=None):
        """Carga los archivos que contiene el set de datos"""
        if cache is None:
            cache = Dataset.PATH_CACHE

        dataset = tf.data.Dataset.list_files(self._filename)
        dataset = dataset.interleave(self.record_dataset, cycle_length=Dataset.CYCLE_LENGTH,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache(filename=path.resolve(cache))
        return dataset

    def load_and_map(self):
        dataset = self.load()
        dataset = dataset.map(self._feature.deserialize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def __call__(self, epoch=1, batch_size=64, buffer_size=1024):
        dataset = self.load()
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(self._feature.deserialize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch_size) if self._feature.padded_shapes() is None \
            else dataset.padded_batch(batch_size, padded_shapes=self._feature.padded_shapes())
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def load_filename(version=None):
        """Set de datos alojado en amazon S3"""
        if version is None:
            version = Dataset.VERSION_NUMBER
        return ['s3://monolieta/dataset/train-{}.tfrecord'.format(version)]
