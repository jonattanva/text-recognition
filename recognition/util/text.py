# coding: utf-8
import tensorflow as tf
import recognition.util.path as path
import recognition.util.utilities as utilities


class Text:
    """Codifica y decodifica un texto"""

    SEPARATOR_EMPTY = " "

    def __init__(self, one_hot=False):
        self._one_hot = one_hot
        self._chars = path.get_chars()
        self._chars_length = len(self._chars)
        self._keys = tf.constant(list(self._chars.keys()), dtype=tf.string)
        self._values = tf.constant(list(self._chars.values()), dtype=tf.int32)

    def encode(self, value, normalize_length=None):
        """Convierte cada letra del texto en su representación numérica"""
        if normalize_length is not None:
            value = Text.normalize_text(value, length=normalize_length)

        if not tf.is_tensor(value):
            if not isinstance(value, list):
                value = [value]
            value = tf.constant(value, dtype=tf.string)

        value_shape = value.get_shape()
        if value_shape.ndims == 0:
            value = tf.expand_dims(value, axis=0)

        string_length = tf.strings.length(value)
        string_length = tf.gather(string_length, tf.math.argmax(string_length, axis=0))
        string_length = tf.cond(tf.equal(string_length, 0), lambda: 1, lambda: string_length)

        batch_size = tf.shape(value)[0]
        shape = [batch_size, string_length, self._chars_length] if self._one_hot else [batch_size, string_length]

        values = utilities.split_by_characters(value)
        values = tf.map_fn(
            lambda key: self.get_code_one_hot(key) if self._one_hot else self.get_value(key), values, dtype=tf.int32)
        values = tf.reshape(values, shape=shape)
        return values

    def get_key(self, value):
        """Obtiene el index del carácter"""
        indices = tf.where(tf.equal(self._values, value))
        return tf.gather(self._keys, indices)

    def get_value(self, key):
        """Obtiene el valor del carácter"""
        value = tf.gather(self._values, self.get_index(key))
        return value

    def get_code_one_hot(self, key):
        """Obtiene el valor tipo 'one-hot' del carácter"""
        indices = self.get_index(key)
        return tf.scatter_nd(indices, [1], shape=[self._chars_length])

    def get_index(self, key):
        """Obtiene el index del carácter"""
        indices = tf.where(tf.equal(self._keys, key))
        indices = tf.cond(tf.equal(tf.size(indices), 0),
                          lambda: tf.where(tf.equal(self._keys, Text.SEPARATOR_EMPTY)), lambda: indices)
        return indices

    def to_sparse_tensor(self, value):
        """Convierte el texto en un 'SparseTensor'"""
        indices = tf.where(tf.math.not_equal(value, self._chars.get(Text.SEPARATOR_EMPTY)))
        values = tf.gather_nd(value, indices)

        return tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=tf.shape(value, out_type=tf.int64))

    @staticmethod
    def normalize_text(value, length, separator=None):
        if separator is None:
            separator = Text.SEPARATOR_EMPTY

        subtract = tf.math.subtract(length, tf.strings.length(value))
        if subtract <= 0:
            return value

        template = tf.fill([1, subtract], value=separator)
        template = tf.reshape(template, shape=[subtract, 1])
        template = tf.strings.reduce_join(template, axis=0)

        value = tf.strings.join([value, template], separator='')
        return value

    @staticmethod
    def resize(text, font, size, **kwargs):
        spacing = kwargs.get('spacing', 4)
        margin = kwargs.get('margin', (50, 50))
        margin_error = kwargs.get('margin_error', 10)

        paragraphs = text.split('\n')
        if len(paragraphs) == 0:
            return text

        pages = []
        new_text = ''  # TODO: Remove line!
        width, height = size
        margin_left, margin_top = margin

        margin_top = margin_top * 2
        margin_left = margin_left * 2

        width = width - margin_left
        height = height - margin_top - spacing

        paragraphs.reverse()
        while paragraphs:
            paragraph = paragraphs.pop()
            if not paragraph:
                new_text = new_text + '\n\n'
                continue

            paragraph_width, _ = font.getsize_multiline(paragraph, spacing=spacing)
            if paragraph_width <= width:
                new_text = new_text + paragraph
                continue

            new_words = ''
            current_width = 0

            words = paragraph.split(' ')
            words.reverse()

            while words:
                word = words.pop()
                word_width, _ = font.getsize(word)

                if (word_width + current_width) >= (width - margin_error):
                    new_words = new_words.rstrip()
                    new_text = new_text + new_words + '\n'
                    new_words = ''

                    _, current_height = font.getsize_multiline(new_text, spacing=spacing)
                    if current_height >= height:
                        print("-----", (current_height, height), "-----")
                        print('>>>>\n', new_text, '\n<<<<<')

                new_words = new_words + word
                if not words and new_words:
                    new_text = new_text + new_words
                    break

                if new_words:
                    new_words = new_words + ' '
                current_width, _ = font.getsize_multiline(new_words, spacing=spacing)

        print('SIZE: ', font.getsize_multiline(new_text, spacing=spacing))
        print('MAX SIZE: ', (width, height))

        print(new_text)
        return new_text
