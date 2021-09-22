import unittest
import numpy as np
import recognition.util.path as path

from recognition.util.text import Text
from recognition.bin.generate import Generate


class GenerateTestCase(unittest.TestCase):

    def setUp(self):
        self._filename_zero = path.resolve('tests/resources/0.txt')
        self._filename_one = path.resolve('tests/resources/1.txt')
        self._filename_five = path.resolve('tests/resources/5.txt')
        self._filename_six = path.resolve('tests/resources/6.txt')

    def test_generate_layer(self):
        generate = Generate({
            Generate.PATH: self._filename_one,
            Generate.LANGUAGE: 'en'
        }, destination_path='temp/test/template0.tfrecord')
        files = generate.read_file()

        font = path.get_fonts()[0]
        pages = generate.get_pages(files, font=font)
        self.assertTrue(len(pages) > 0)

        page = pages[0].get('page', '')
        language = pages[0].get('language', None)
        image, boxes = generate.generate_layer(page, font=font, language=language)

        self.assertEqual(8, len(boxes))
        self.assertTrue(np.array_equal([
            [48, 48, 339, 84, 1],
            [48, 96, 137, 132, 1],
            [48, 176, 350, 212, 1],
            [48, 224, 171, 260, 1],
            [48, 272, 414, 308, 1],
            [48, 320, 187, 356, 1],
            [48, 368, 365, 420, 1],
            [48, 448, 391, 628, 1]], boxes))

    def test_get_pages(self):
        generate = Generate({
            Generate.PATH: self._filename_one,
            'language': 'en'
        }, 'temp/test/template0.tfrecord')
        files = generate.read_file()

        font = path.get_fonts()[0]
        pages = generate.get_pages(files, font=font)
        for page in pages:
            self.assertFalse(page.get('page', '')[0] == '')
            self.assertEqual('en', page.get('language'))

        pages = generate.get_pages([], font=font)
        self.assertTrue(len(pages) == 0)

    def test_check_size_page(self):
        generate = Generate({
            Generate.PATH: self._filename_zero,
            'language': 'es'
        }, 'temp/test/template0.tfrecord')
        files = generate.read_file()

        font = path.get_fonts()[0]
        pages = generate.get_pages(files, font=font)

        after_width, after_height = font.getsize_multiline(pages[0].get('page'))
        before_width, before_height = font.getsize_multiline(files[0].get('text'))

        self.assertTrue(after_width <= before_width)
        self.assertTrue(after_height >= before_height)

    def test_read_file(self):
        generate = Generate({
            Generate.PATH: self._filename_one
        }, 'temp/test/template0.tfrecord')
        files = generate.read_file()

        self.assertEqual('1.txt', files[0].get('name'))
        self.assertEqual(path.load_txt(self._filename_one), files[0].get('text'))
        self.assertIsNone(files[0].get('language'))

    def test_error_read_file(self):
        generate = Generate({
            Generate.PATH: ''
        }, 'temp/test/template0.tfrecord')
        self.assertRaises(ValueError, generate.read_file)

    def test_writer_extract_text(self):
        generate = Generate({
            Generate.PATH: self._filename_one,
            Generate.LANGUAGE: 'en'
        }, 'temp/test/template0.tfrecord')
        generate.writer_extract_text()
        self.assertTrue(path.exists(path.resolve('temp/test/template0.tfrecord')))

    def test_text_fill(self):
        generate = Generate({
            Generate.PATH: self._filename_five,
            Generate.LANGUAGE: 'es'
        }, destination_path='temp/test/template5.tfrecord')

        font = path.get_fonts()[0]
        files = generate.read_file()
        pages = generate.get_pages(files, font=font)

        page = pages[0].get('page', '')
        language = pages[0].get('language', None)
        image, boxes = generate.generate_layer(page, font=font, language=language)

        self.assertEqual(1, len(boxes))
        self.assertTrue(np.array_equal([[48, 48, 452, 228, 2]], boxes))

    def test_apply_margin(self):
        generate = Generate({
            Generate.PATH: self._filename_five,
            Generate.LANGUAGE: 'es'
        }, destination_path='temp/test/template5.tfrecord')

        margin = generate.apply_margin()
        self.assertEqual((476, 668), margin)

        margin = generate.apply_margin(margin=(100, 100))
        self.assertEqual((376, 568), margin)

    def test_resize_text(self):
        generate = Generate({
            Generate.PATH: self._filename_six,
            Generate.LANGUAGE: 'es'
        }, destination_path='temp/test/template_6.tfrecord')

        files = generate.read_file()
        text = files[0].get("text")

        font = path.get_fonts()[0]
        text = Text.fill(text, font=font, width=376)
        # text = generate.resize_text(text, font=font)

        # self.assertEqual((384, 149), font.getsize_multiline(text, spacing=4))

    if __name__ == '__main__':
        unittest.main()
