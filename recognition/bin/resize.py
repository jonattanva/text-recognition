# coding: utf-8
import argparse
import recognition.util.path as path

from recognition.util.text import Text


class Resize:

    def __init__(self, filename, destination_path, **kwargs):
        self._filename = filename
        self._destination_path = destination_path
        self._size = kwargs.get('size', (576, 768))

    def generate(self):
        fonts = path.get_fonts()
        if len(fonts) == 0:
            raise ValueError('No font found')

        text = path.load_txt(self._filename)
        for font in fonts:
            pages = Text.resize(text, font=font, size=self._size)
            with open(self._destination_path, 'w') as f:
                for page in pages:
                    f.write(page)


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename',
                        type=str,
                        default=None,
                        metavar='',
                        help='Ruta del archivo')

    parser.add_argument('--destination-path',
                        type=str,
                        metavar='',
                        default=None,
                        help='Ruta donde se guarda el archivo de resultado')

    parser.add_argument('--width',
                        type=int,
                        default=576,
                        metavar='',
                        help="Ancho de las imágenes")

    parser.add_argument('--height',
                        type=int,
                        default=768,
                        metavar='',
                        help="Alto de las imágenes")

    return parser.parse_args()


if __name__ == '__main__':
    arguments = process_args()

    if not arguments.filename:
        raise ValueError('The filename is undefined')

    if not arguments.destination_path:
        raise ValueError('The destination path is undefined')

    resize = Resize(
        filename=arguments.filename,
        destination_path=arguments.destination_path,
        size=(arguments.width, arguments.height))

    resize.generate()
