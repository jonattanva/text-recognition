import unittest
import PIL.Image
import numpy as np
import recognition.util.path as path
import recognition.util.image

from recognition.util.text import Text


class TokenTestCase(unittest.TestCase):

    def setUp(self):
        self.text = Text()

    def test_encode(self):
        self.assertTrue(np.array_equal([[5, 14, 3, 16, 4, 5]], self.text.encode('encode').numpy()))
        self.assertTrue(np.array_equal([[8, 9]], self.text.encode('hi').numpy()))
        self.assertTrue(
            np.array_equal([[8, 5, 12, 12, 16, 0, 24, 16, 19, 12, 4]], self.text.encode('hello world').numpy()))

    def test_adjust_text(self):
        font = path.get_fonts()[0]
        sample_text = path.load_txt(path.resolve('tests/resources/7.txt'))

        pages = Text.resize(sample_text, font=font, size=(576, 768))

        """
        width, height = font.getsize_multiline(pages)
        print("size", (width, height))

        image = PIL.Image.new('RGB', size=(576, 768), color=(255, 255, 255))
        recognition.util.image.draw_text(image, pages[0], font, fill='#333333', align='left')

        # self.assertEqual(len(pages), 3)

        # page = pages[0].split('\n')
        # self.assertEqual(len(page), 12)
        """

    def test_adjust_text_height(self):
        font = path.get_fonts()[0]
        sample_text = """The textwrap module can be used 
        to format text for output in situations where 
        pretty-printing is desired. It offers programmatic 
        functionality similar to the paragraph wrapping 
        or filling features found in many text editors."""
        pages = Text.resize(sample_text, font=font, size=(400, 200))
        self.assertEqual(len(pages), 1)


if __name__ == '__main__':
    unittest.main()
