import unittest
import tensorflow as tf
from recognition.model.darknet.layer.prepare import Prepare


class PrepareTestCase(unittest.TestCase):

    def test_model_shape(self):
        prepare = Prepare(filters=256, number_anchors=3, number_classes=1)
        result = prepare(tf.zeros([48, 52, 52, 128]), training=False)
        self.assertEqual([48, 52, 52, 3, 6], result.shape)

        prepare = Prepare(filters=128, number_anchors=3, number_classes=10)
        result = prepare(tf.zeros([48, 52, 52, 128]), training=True)
        self.assertEqual([48, 52, 52, 3, 15], result.shape)

    def test_name(self):
        prepare = Prepare(filters=256)
        self.assertEqual('prepare', prepare.name)

        prepare = Prepare(filters=256, name='prepare_test')
        self.assertEqual('prepare_test', prepare.name)

    def test_weights(self):
        inputs = tf.zeros([48, 52, 52, 128])

        prepare = Prepare(filters=128, number_anchors=3, number_classes=1)
        prepare(inputs, training=True)

        self.assertEqual(7, len(prepare.weights))
        self.assertEqual(5, len(prepare.trainable_weights))
        self.assertEqual(2, len(prepare.non_trainable_weights))

    def test_config(self):
        prepare = Prepare(filters=256, number_anchors=3, number_classes=1)
        config = prepare.get_config()
        self.assertEqual({
            'name': 'prepare',
            'trainable': True,
            'dtype': 'float32',
            'filters': 256,
            'number_anchors': 3,
            'number_classes': 1
        }, config)


if __name__ == '__main__':
    unittest.main()
