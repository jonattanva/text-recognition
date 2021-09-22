# coding: utf-8
import time
import argparse
import PIL.Image
import progress.bar
import numpy as np
import tensorflow as tf
import recognition.util.image
import recognition.util.path as path
import recognition.bin.get_kmeans as kmeans

from recognition.parse.feature import Feature
from recognition.service.dataset import Dataset


class Generate:
    PATH = "path"
    PAGE = "page"
    BBOX = "bbox"
    NAME = "name"
    TEXT = "text"
    SIZE = "size"
    IMAGE = "image"
    LABEL = "label"
    LANGUAGE = 'language'
    ALIGNMENT = 'alignment'
    ALIGNMENT_LEFT = 'left'

    EXTENSION_CSV = ".csv"
    EXTENSION_TXT = ".txt"
    EXTENSION_JSON = ".json"

    COLOR_FILL = [
        "#818181", "#161616", "#292929", "#3d3d3d", "#505050", "#646464", "#787878",
        "#959595", "#a9a9a9", "#bcbcbc", "#d0d0d0"
    ]

    def __init__(self, filename, destination_path, **kwargs):
        if not destination_path:
            raise ValueError('The destination path is empty')

        if not isinstance(filename, list):
            filename = [filename]

        self._filename = filename
        self._base_path = kwargs.get('base_path', None)
        self._classes = path.get_class_name()
        self._destination_path = destination_path
        self._size = kwargs.get('size', (576, 768))
        self._compression_type = kwargs.get('compression_type', Dataset.COMPRESSION_TYPE)

    def read_file(self):
        """Lee los archivos según su extensión"""
        files = []
        for filename in self._filename:
            file_path = filename.get(Generate.PATH)
            language = filename.get(Generate.LANGUAGE, None)
            alignment = filename.get(Generate.ALIGNMENT, Generate.ALIGNMENT_LEFT)

            if self._base_path is not None:
                file_path = self._base_path + file_path

            callback = Generate.load_file(extension=path.extension(file_path))
            text = callback(file_path)

            if not text:
                continue

            files.append({
                Generate.NAME: path.basename(file_path),
                Generate.TEXT: text,
                Generate.LANGUAGE: language,
                Generate.ALIGNMENT: alignment
            })

        return files

    def generate_layer(self, text, font, **kwargs):
        """Genera la imagen con el texto y sus cajas delimitadoras"""
        language = kwargs.get('language', None)
        color = kwargs.get('color', (255, 255, 255))
        align = kwargs.get('align', Generate.ALIGNMENT_LEFT)

        fill = np.random.choice(Generate.COLOR_FILL)

        image = PIL.Image.new('RGB', size=self._size, color=color)
        recognition.util.image.draw_text(image, text, font, fill=fill, align=align)

        language = self._classes.get(language)
        boxes = recognition.util.image.generate_boxes(
            image, text, font, language=language, align=align)

        return image, boxes

    def apply_margin(self, margin=(50, 50)):
        """Obtiene el tamaño real de la imagen descontando el margen"""
        width, height = self._size
        margin_left, margin_top = margin

        max_width = width - (margin_left * 2)
        max_height = height - (margin_top * 2)

        return max_width, max_height

    def get_pages(self, files, font, margin=(50, 50), spacing=4):
        total_files = len(files)
        if total_files == 0:
            return []

        pages = []
        max_width, max_height = self.apply_margin(margin=margin)

        bar = progress.bar.Bar('Read files', max=total_files)
        for file in files:
            text = file.get(Generate.TEXT, '')
            if len(text) == 0:
                continue

            page = ''
            lines = text.split('\n')
            number_lines = len(lines)

            language = file.get(Generate.LANGUAGE)
            alignment = file.get(Generate.ALIGNMENT)

            for step, line in enumerate(lines):
                if not page and not line:
                    continue

                include_page = False
                next_step = step + 1
                if next_step < number_lines:
                    _, next_height = font.getsize_multiline(page + lines[next_step] + '\n', spacing=spacing)
                    include_page = next_height >= max_height

                if not include_page:
                    _, current_height = font.getsize_multiline(page, spacing=spacing)
                    include_page = current_height >= max_height

                if not include_page and step == (number_lines - 1):
                    include_page = True
                    page = page + line + '\n'

                if include_page:
                    pages.append({
                        Generate.PAGE: page,
                        Generate.LANGUAGE: language,
                        Generate.ALIGNMENT: alignment
                    })

                    page = ''
                    continue

                page = page + line + '\n'

            bar.next()

        bar.finish()
        return pages

    def writer_extract_text(self, generate_kmeans=False):
        extract_text = self.read_file()
        if len(extract_text) == 0:
            raise ValueError('Error extract text from file')

        fonts = path.get_fonts()
        if len(fonts) == 0:
            raise ValueError('No font found')

        results = []
        start = time.time()
        location_bbox_max_length = 0
        for font in fonts:
            pages = self.get_pages(extract_text, font=font)
            bar = progress.bar.Bar('Generating', max=len(pages))

            for step, page in enumerate(pages):
                label = page.get(Generate.PAGE)
                language = page.get(Generate.LANGUAGE, None)
                alignment = page.get(Generate.ALIGNMENT, Generate.ALIGNMENT_LEFT)
                image, bbox = self.generate_layer(label, font=font, language=language, align=alignment)

                location_bbox_max_length = max(location_bbox_max_length, len(bbox))
                content = recognition.util.image.to_bytes(image)

                label = label.encode(encoding='utf-8')
                results.append({
                    Generate.IMAGE: content,
                    Generate.LABEL: label,
                    Generate.SIZE: (image.height, image.width),
                    Generate.BBOX: bbox,
                    Generate.ALIGNMENT: alignment
                })

                image.close()
                bar.next()

            bar.finish()

        feature = Feature()
        destination_path = path.resolve(self._destination_path)
        options = tf.io.TFRecordOptions(compression_type=self._compression_type)

        path.create_folder_if_not_exists(destination_path)
        with tf.io.TFRecordWriter(destination_path, options=options) as writer:
            bar = progress.bar.Bar('Writing', max=len(results))
            for result in results:
                writer.write(feature.serialize(
                    result[Generate.IMAGE],
                    result[Generate.LABEL],
                    size=result[Generate.SIZE],
                    bbox=result[Generate.BBOX],
                    bbox_max_length=location_bbox_max_length,
                    alignment=result[Generate.ALIGNMENT]
                ))

                writer.flush()
                bar.next()

            bar.finish()

        print('Time {} sec'.format(time.time() - start))
        print('Output saved to: {}'.format(destination_path))

        if generate_kmeans:
            boxes_result = kmeans.get_boxes(
                filename=destination_path, compression_type=Dataset.COMPRESSION_TYPE)

            anchors_result, ave_iou_result = kmeans.get_kmeans(boxes_result, cluster_number=kmeans.CLUSTER_NUMBER)
            anchor_string = kmeans.anchors_prepare(anchors_result)

            print('Anchors are: {}'.format(anchor_string))
            print('The average iou is: {}'.format(ave_iou_result))

    @staticmethod
    def prepare_filename(filename):
        extension = path.extension(filename)

        if extension == Generate.EXTENSION_CSV:
            content = path.load_csv(filename)
            filename = [{
                Generate.PATH: row[0],
                Generate.LANGUAGE: row[1],
                Generate.ALIGNMENT: row[2]
            } for row in content]

        elif extension == Generate.EXTENSION_JSON:
            content = path.load_json(filename)
            filename = content.get('datasources', [])

        return filename

    @staticmethod
    def load_file(extension):
        switcher = {
            Generate.EXTENSION_TXT: path.load_txt,
            Generate.EXTENSION_CSV: path.load_csv,
            Generate.EXTENSION_JSON: path.load_json
        }

        def invalid(value):
            raise ValueError("The file has an invalid extension {}".format(value))

        return switcher.get(extension, lambda _: invalid(extension))


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename',
                        type=str,
                        default=None,
                        metavar='',
                        help='')

    parser.add_argument('--destination-path',
                        type=str,
                        metavar='',
                        default=None,
                        help='')

    parser.add_argument('--base-path',
                        type=str,
                        default=None,
                        metavar='',
                        help='')

    parser.add_argument('--width',
                        type=int,
                        default=576,
                        metavar='',
                        help="")

    parser.add_argument('--height',
                        type=int,
                        default=768,
                        metavar='',
                        help="")

    parser.add_argument('--compression-type',
                        type=str,
                        metavar='',
                        default='GZIP',
                        help='')

    parser.add_argument('--generate-kmeans',
                        type=bool,
                        metavar='',
                        default=True,
                        help='')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = process_args()

    if not arguments.filename:
        raise ValueError('The filename is undefined')

    if not arguments.destination_path:
        raise ValueError('The destination path is undefined')

    arguments.filename = Generate.prepare_filename(filename=arguments.filename)

    generate = Generate(
        filename=arguments.filename,
        destination_path=arguments.destination_path,
        base_path=arguments.base_path,
        size=(arguments.width, arguments.height),
        compression_type=arguments.compression_type)

    generate.writer_extract_text(
        generate_kmeans=arguments.generate_kmeans)
