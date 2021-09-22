# coding: utf-8
import tensorflow as tf


def get_key_from_dictionary(value, dictionary):
    """Obtiene la llave del elemento que esta en diccionario"""
    return list(dictionary.keys())[list(dictionary.values()).index(value)]


def split_by_characters(value):
    """Separa una cadena de texto por caracteres"""

    def condition(step, _):
        return tf.math.less(step, tf.strings.length(value)[0])

    def body(step, output):
        ref = tf.strings.substr(value, pos=step, len=1)
        output = output.write(step, ref)
        return step + 1, output

    start = tf.constant(0)
    array = tf.TensorArray(tf.string, size=0, dynamic_size=True)

    _, values = tf.while_loop(condition, body, loop_vars=[start, array])
    values = values.stack()
    return values
