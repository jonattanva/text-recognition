import unittest
import numpy as np

import recognition.util.path as path


class PathTestCase(unittest.TestCase):

    def test_listdir(self):
        self.assertTrue(len(path.listdir('tests')) > 0)

    def test_dirname(self):
        self.assertEqual('recognition/util', path.dirname('recognition/util/path.py'))

    def test_extension(self):
        self.assertEqual('.py', path.extension('recognition/util/path.py'))

    def test_exists(self):
        self.assertTrue(path.exists(path.resolve('tests')))
        self.assertFalse(path.exists(path.resolve('fake')))

    def test_get_anchors(self):
        anchors = path.get_anchors()
        self.assertTrue(np.array_equal([9, 2], anchors.shape))

    def test_get_chars(self):
        chars = path.get_chars()
        self.assertTrue(len(chars) > 0)

    def test_get_class_name(self):
        class_name = path.get_class_name()
        self.assertTrue(len(class_name) > 0)

    def test_name(self):
        self.assertEqual('recognition/util/path', path.name('recognition/util/path.py'))

    def test_basename(self):
        self.assertEqual('path.py', path.basename('recognition/util/path.py'))

    def test_get_fonts(self):
        fonts = path.get_fonts()
        self.assertTrue(len(fonts) > 0)

    def test_create_folder(self):
        path_folder = path.resolve('temp/test/path')
        path.create_folder(path_folder)

        self.assertTrue(path.exists(path_folder))


if __name__ == '__main__':
    unittest.main()
