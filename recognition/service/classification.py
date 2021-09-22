# coding: utf-8
import tensorflow as tf
import recognition.util.path as path

from recognition.util.text import Text
from recognition.model.decode import Decode
from recognition.service.training import Training
from recognition.model.vision.vision import Vision


class Classification(Training):

    def __init__(self, strategy=None, destination_path=None, decode_type=Decode.BEST_PATH):
        self._chars = path.get_chars()
        self._decode_type = decode_type
        self._text = Text(one_hot=False)
        super(Classification, self).__init__(
            model=Vision(chars=self._chars), strategy=strategy, destination_path=destination_path)

    def loss_step(self, labels, predictions):
        labels = self._text.to_sparse_tensor(labels)
        loss_value = tf.nn.ctc_loss(
            labels=labels,
            logits=predictions,
            label_length=None,
            logit_length=[len(predictions)],
            logits_time_major=True,
            blank_index=-1)

        loss_value = tf.math.add(loss_value, tf.math.reduce_sum(self._model.losses))
        return loss_value

    def accuracy_step(self, labels, predictions):
        decoded = self.decode(predictions)
        decoded = tf.cast(decoded, dtype=tf.int32)
        return labels, decoded

    def decode(self, predictions):
        switcher = {
            Decode.BEST_PATH: Classification.greedy_decoder,
            Decode.BEAM_SEARCH: Classification.beam_search_decoder,
        }

        callback = switcher.get(self._decode_type)
        decoded = callback(predictions)

        decoded = tf.reshape(decoded.values, shape=decoded.dense_shape)
        return decoded

    @staticmethod
    def greedy_decoder(inputs, merge_repeated=True):
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs, sequence_length=[len(inputs)], merge_repeated=merge_repeated)
        decoded = decoded[0]
        return decoded

    @staticmethod
    def beam_search_decoder(inputs, top_paths=1, beam_width=100):
        decoded, _ = tf.nn.ctc_beam_search_decoder(
            inputs, sequence_length=[len(inputs)], beam_width=beam_width, top_paths=top_paths)
        decoded = decoded[0]
        return decoded
