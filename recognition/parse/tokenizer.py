# coding: utf-8
from recognition.util.text import Text
from recognition.parse.feature import Feature


class Tokenizer(Feature):

    def __init__(self, one_hot=False, channel=None):
        super(Tokenizer, self).__init__(channel=channel)
        self._text = Text(one_hot=one_hot)

    def deserialize(self, serialized):
        images, label, extra = super(Tokenizer, self).deserialize(serialized)
        normalize_length = extra[Feature.CLASSIFICATION_LABEL_MAX_LENGTH]

        encode = self._text.encode(label, normalize_length=normalize_length)
        return images, encode

    def padded_shapes(self):
        return None
