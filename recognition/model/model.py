# coding: utf-8
import tensorflow as tf
import recognition.util.path as path


class Model(tf.keras.Model):
    PATH_WEIGHT = "model/{}.h5"

    def __init__(self, name):
        super(Model, self).__init__(name=name)

    def save_weights(self, filepath, overwrite=True, save_format=None):
        if not path.exists(path.dirname(filepath)):
            path.create_folder(path.dirname(filepath))

        super(Model, self).save_weights(
            filepath, overwrite=overwrite, save_format=save_format)

    def load_weights(self, filepath=None, by_name=False, skip_mismatch=False):
        if filepath is None:
            filepath = path.resolve(Model.PATH_WEIGHT.format(self.name))

        if not path.exists(filepath):
            raise ValueError('the weight model does not exist')

        print('Load the model: {}'.format(filepath))
        super(Model, self).load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch)
