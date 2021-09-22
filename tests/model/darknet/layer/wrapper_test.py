import unittest

from recognition.model.darknet.layer.wrapper import Wrapper


class WrapperTestCase(unittest.TestCase):

    def test_name(self):
        wrapper = Wrapper(256)
        self.assertEqual('wrapper', wrapper.name)

        wrapper = Wrapper(128, name='wrapper_test')
        self.assertEqual('wrapper_test', wrapper.name)

    def test_config(self):
        wrapper = Wrapper(512)
        config = wrapper.get_config()
        self.assertEqual({
            'dtype': 'float32',
            'filters': 512,
            'name': 'wrapper',
            'trainable': True
        }, config)


if __name__ == '__main__':
    unittest.main()
