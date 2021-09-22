import unittest

from recognition.model.vision.layer.bidirectional import Bidirectional


class BidirectionalTestCase(unittest.TestCase):

    def test_name(self):
        bidirectional = Bidirectional()
        self.assertEqual("bidirectional", bidirectional.name)

        bidirectional = Bidirectional(name="bidirectional_test")
        self.assertEqual("bidirectional_test", bidirectional.name)

    def test_config(self):
        bidirectional = Bidirectional()
        config = bidirectional.get_config()
        self.assertEqual({
            'name': 'bidirectional',
            'trainable': True,
            'dtype': 'float32',
            'units': 128
        }, config)


if __name__ == '__main__':
    unittest.main()
