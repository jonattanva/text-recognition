import unittest
import recognition.util.path as path

from recognition.bin.generate import Generate
from recognition.parse.feature import Feature
from recognition.service.dataset import Dataset


class DatasetTestCase(unittest.TestCase):

    def setUp(self):
        generate = Generate({
            'path': path.resolve('tests/resources/4.txt'),
            'language': 'en'
        }, 'temp/template1.tfrecord')
        generate.writer_extract_text()
        self._feature = Feature(channel=3)
        self._filename = path.resolve('temp/template1.tfrecord')

    def test_load(self):
        self.assertTrue(path.exists(self._filename))

        dataset = Dataset(self._filename, self._feature)
        dataset = dataset.load()

        self.assertEqual(2, len(list(dataset)))

    def test_error_load(self):
        self.assertRaises(ValueError, Dataset, None, None, None)

    def test_call(self):
        dataset = Dataset(self._filename, self._feature)
        dataset = dataset(batch_size=1)

        self.assertEqual(2, len(list(dataset)))

    def test_load_filename(self):
        filename = Dataset.load_filename()
        self.assertTrue(len(filename) > 0)


if __name__ == '__main__':
    unittest.main()
