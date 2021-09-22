# coding: utf-8
import re
import io
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw
import tensorflow as tf
import matplotlib.pyplot as plt
import recognition.util.path as path
import recognition.util.utilities as utilities

OPTIMAL_FONT_SIZE = "hg"


def resize(image, size):
    """Cambia el tamaño de la imagen al ancho y alto indicado manteniendo la relación de aspecto"""
    target_height, target_width = size
    image = tf.image.resize_with_pad(image, target_height=target_height, target_width=target_width)
    return image


def resize_image_and_boxes(images, size, boxes):
    """Cambia el tamaño de la imagen al ancho y alto indicado manteniendo la relación de aspecto
    además ajusta todos los cuadros delimitadores a la nueva resolución"""
    if not tf.is_tensor(boxes):
        boxes = tf.constant(boxes, dtype=tf.float32)

    image_shape = tf.shape(images)
    original_height, original_width = tf.cond(tf.math.equal(tf.size(image_shape), 4),
                                              lambda: (image_shape[1], image_shape[2]),
                                              lambda: (image_shape[0], image_shape[1]))
    target_height, target_width = size
    images = resize(images, (target_height, target_width))

    scale = tf.math.minimum(tf.math.divide(target_height, original_height),
                            tf.math.divide(target_width, original_width))
    scale = tf.cast(scale, dtype=tf.float32)

    new_height = tf.math.multiply(scale, tf.cast(original_height, dtype=tf.float32))
    new_height = tf.cast(new_height, dtype=tf.int32)

    new_width = tf.math.multiply(scale, tf.cast(original_width, dtype=tf.float32))
    new_width = tf.cast(new_width, dtype=tf.int32)

    left = tf.math.divide(tf.math.subtract(target_width, new_width), 2)
    left = tf.cast(left, dtype=tf.float32)

    upper = tf.math.divide(tf.math.subtract(target_height, new_height), 2)
    upper = tf.cast(upper, dtype=tf.float32)

    boxes_shape = boxes.get_shape()
    boxes = tf.cond(tf.math.equal(boxes_shape.ndims, 2), lambda: tf.expand_dims(boxes, axis=0), lambda: boxes)

    boxes = tf.map_fn(lambda values: tf.map_fn(
        lambda value: resize_boxes(value, xy=(left, upper), scale=scale), values), boxes)
    return images, boxes


def resize_boxes(box, xy, scale):
    """Cambia el tamaño de los cuadros delimitadores a la escala indicada"""
    if tf.math.reduce_sum(box[0:4]) == 0:
        return box

    left, upper = xy
    x_min, y_min, x_max, y_max, classes = box[0], box[1], box[2], box[3], box[4]

    x_min = (x_min * scale) + left
    y_min = (y_min * scale) + upper
    x_max = (x_max * scale) + left
    y_max = (y_max * scale) + upper

    box = tf.stack([x_min, y_min, x_max, y_max, classes], axis=-1)
    return box


def crop(image, box, size):
    """Corta y cambia el tamaño de la imagen"""
    height, width = size
    x_min, y_min, x_max, y_max = box

    image = image[y_min:y_max, x_min:x_max, :]
    image = tf.image.resize_with_crop_or_pad(image, width, height)
    return image


def to_bytes(image, format_file='JPEG'):
    """Convierte una imagen en bytes"""
    with io.BytesIO() as output:
        image.save(output, format=format_file)
        content = output.getvalue()
    return content


def to_decode(image, shape):
    """Convierte la imagen de tipo bytes a un tensor de tipo float32"""
    height, width, channel = shape
    image = tf.image.decode_image(image, channels=channel, dtype=tf.float32)
    image = tf.reshape(image, shape=(height, width, channel))
    return image


def draw_text(image, text, font, **kwargs):
    """Escribe un texto sobre la imagen"""
    if not isinstance(image, PIL.Image.Image):
        raise ValueError('the image are not an instance of PIL.Image.Image')

    spacing = kwargs.get('spacing', 4)
    align = kwargs.get('align', 'left')
    fill = kwargs.get('fill', (90, 90, 90))
    margin = kwargs.get('margin', (50, 50))

    margin_left, margin_top = margin
    target_width, target_height = image.size

    target_width = target_width - (margin_left * 2)
    target_height = target_height - (margin_top * 2)

    image_draw = PIL.ImageDraw.Draw(image)
    text_width, text_height = image_draw.multiline_textsize(text, font=font, spacing=spacing)

    if text_width > target_width:
        raise ValueError('the text is very wide {}px - {}px\n\n\n{}'.format(text_width, target_width, text))

    if text_height > target_height:
        raise ValueError('the text is very large {}px - {}px'.format(text_height, target_height))

    lines = text.split('\n')
    if len(lines) == 0:
        return

    x_min, y_min = 0, margin_top
    _, height = font.getsize(OPTIMAL_FONT_SIZE)  # Se obtiene el alto de la fuente

    for step, line in enumerate(lines):
        width, _ = image_draw.textsize(line, font=font, spacing=spacing)
        width_strip, height_strip = image_draw.textsize(line.strip(), font=font, spacing=spacing)

        x_min = width - width_strip  # left
        if align == 'center':
            x_min = (target_width - width) / 2

        if align == 'right':
            x_min = target_width - width

        x_min = x_min + margin_left
        image_draw.text((x_min, y_min), line, font=font, fill=fill)
        y_min = y_min + height


def generate_boxes(image, text, font, **kwargs):
    """Genera la cajas delimitadoras del texto"""
    align = kwargs.get('align', 'left')
    margin = kwargs.get('margin', (50, 50))
    padding = kwargs.get('padding', (2, 2))
    language = kwargs.get('language', None)

    margin_left, margin_top = margin
    padding_left, padding_top = padding
    target_width, target_height = image.size

    target_width = target_width - (margin_left * 2)

    lines = text.split('\n')
    number_lines = len(lines)
    if number_lines > 0 and lines[-1]:
        lines.append('')

    if language is None:
        language = 0

    start = 0
    boxes = []
    y_min = margin_top
    _, height = font.getsize(OPTIMAL_FONT_SIZE)  # Se obtiene el alto de la fuente

    regular_expression = r'[-*]'
    for step, line in enumerate(lines):
        line = re.sub(regular_expression, '', line).strip()
        if line:
            continue

        next_line = step + 1
        if next_line < number_lines:
            line = re.sub(regular_expression, '', lines[next_line]).strip()
            if not line:
                continue

        current_paragraph = lines[start:step]
        paragraph = '\n'.join(current_paragraph)
        if not paragraph:
            continue

        paragraph_flatten_width, paragraph_flatten_height = 0, 0
        for current in current_paragraph:
            current = current.strip()
            flatten_width, _ = font.getsize(current)
            paragraph_flatten_width = max(flatten_width, paragraph_flatten_width)
            if current:
                paragraph_flatten_height = paragraph_flatten_height + height

        paragraph_width, _ = font.getsize_multiline(paragraph)
        paragraph_height = len(current_paragraph) * height

        paragraph_width_adjust = paragraph_width - paragraph_flatten_width
        if align == 'center':
            paragraph_width_adjust = (target_width - paragraph_width) / 2

        if align == 'right':
            paragraph_width_adjust = target_width - paragraph_width

        paragraph_width_adjust = paragraph_width_adjust if paragraph_width_adjust > 0 else 0

        paragraph_height_adjust = paragraph_height - paragraph_flatten_height
        paragraph_height_adjust = paragraph_flatten_height if paragraph_height_adjust > 0 else paragraph_height

        x_min = paragraph_width_adjust + margin_left
        x_max = x_min + paragraph_width
        y_max = y_min + paragraph_height_adjust

        boxes.append([
            x_min - padding_left,
            y_min - padding_top,
            x_max + padding_left,
            y_max + padding_top, language])

        start = step + 1
        y_min = y_min + paragraph_height + height

    return boxes


def draw(images, boxes, **kwargs):
    """Incluye las cajas delimitadoras a la imagen"""
    if len(boxes) == 0:
        return images

    classes = kwargs.get('classes', None)
    normalize = kwargs.get('normalize', False)

    result = []
    images, boxes = prepare_image_and_boxes(images, boxes)

    font = PIL.ImageFont.load_default()
    batch_size, height, width, channel = tf.shape(images)

    for i in range(batch_size):
        image = images[i]
        image = from_array(image.numpy() * 255)
        draw_image = PIL.ImageDraw.Draw(image)

        for k in range(tf.shape(boxes[i])[0]):
            x_min, y_min, x_max, y_max, key_class = boxes[i][k]
            x_min = tf.cast(x_min, dtype=tf.int32)
            y_min = tf.cast(y_min, dtype=tf.int32)
            x_max = tf.cast(x_max, dtype=tf.int32)
            y_max = tf.cast(y_max, dtype=tf.int32)

            if normalize:
                x_min = x_min * width
                y_min = y_min * height
                x_max = x_max * width
                y_max = y_max * height

            draw_image.rectangle(((x_min, y_min), (x_max, y_max)), width=1, outline="#7b06ff")
            if classes is not None:
                key_value = utilities.get_key_from_dictionary(key_class, classes)
                key_value = key_value.upper()
                font_width, font_height = font.getsize(key_value)

                xy = (x_min, y_min), (x_min + font_width + 2, y_min - font_height)
                draw_image.rectangle(xy, fill="#7b06ff")

                xy = (x_min + 2, y_min - font_height)
                draw_image.text(xy, key_value, fill="#fff")

        image = to_bytes(image)
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        result.append(image)

    result = tf.reshape(result, shape=[batch_size, height, width, channel])
    return result


def from_array(image, dtype='uint8'):
    """Convierte un numpy array en una imagen"""
    return PIL.Image.fromarray(image.astype(dtype), 'RGB')


def save(images, filename):
    """Convierte la imagen y la guarda"""
    path.create_folder_if_not_exists(filename)

    if not tf.is_tensor(images):
        images = tf.convert_to_tensor(images)

    images_shape = images.get_shape()
    if images_shape.ndims == 3:
        images = tf.expand_dims(images, axis=0)

    batch_size = tf.shape(images)[0]
    for i in range(batch_size):
        image = from_array(images[i].numpy())
        image.save(filename)
        image.close()


def from_file(filename):
    """Carga la imagen desde un archivo"""
    with open(filename, mode='rb') as f:
        return PIL.Image.open(io.BytesIO(f.read()))


def crop_and_resize(images, boxes):
    """Obtiene la imagen que corresponda a su caja delimitadora"""
    result = []
    images, boxes = prepare_image_and_boxes(images, boxes)

    batch_size = tf.shape(images)[0]
    height, width = get_max_size_boxes(boxes)

    for i in range(batch_size):
        image = images[i]
        image_boxes = tf.map_fn(lambda value: crop(
            image, box=(value[0], value[1], value[2], value[3]), size=(height, width)
        ), boxes[i], dtype=tf.float32)
        result.append((image, image_boxes))

    return result


def get_max_size_boxes(boxes):
    """Obtiene el tamaño máximo de las cajas delimitadoras"""
    size_boxes = get_size_boxes(boxes)

    max_size = tf.math.reduce_max(size_boxes, axis=1)
    max_size = tf.reshape(max_size, shape=[-1])
    return max_size


def get_size_boxes(boxes):
    """Obtiene el tamaño de cada caja delimitadora"""
    if not tf.is_tensor(boxes):
        boxes = tf.convert_to_tensor(boxes)

    boxes_shape = boxes.get_shape()
    if boxes_shape.ndims == 2:
        boxes = tf.expand_dims(boxes, axis=0)

    return tf.math.subtract(boxes[..., 2:4], boxes[..., 0:2])


def prepare_image_and_boxes(images, boxes):
    images_shape = images.get_shape()
    if images_shape.ndims == 3:
        images = tf.expand_dims(images, axis=0)

    if not tf.is_tensor(boxes):
        boxes = tf.convert_to_tensor(boxes)

    boxes_shape = boxes.get_shape()
    if boxes_shape.ndims == 2:
        boxes = tf.expand_dims(boxes, axis=0)

    return images, boxes


def plot_to_image(figure):
    """Convierte el gráfico matplotlib en una imagen PNG"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, axis=0)
    return image
