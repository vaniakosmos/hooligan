import easy_flags
import tensorflow as tf
from tensorflow.contrib import rnn


class ModelConfig(easy_flags.BaseConfig):
    layers = 2
    cell_size = 128
    forget_bias = 1.0
    input_keep_prob = 1.0
    output_keep_prob = 1.0
    time_steps = 281  # tweet size + break char
    vocab_size = 64
    grad_clip = 5
    batch_size = 64
    vocab_path = '../data/vocab.json'


class Model(object):
    def __init__(self, c: ModelConfig, training=True):
        self.training = training
        if not self.training:
            c.batch_size = 1
            c.time_steps = 1

        self.inputs_ph = tf.placeholder(tf.int64, [c.batch_size, c.time_steps],
                                        name='inputs_ph')
        self.targets_ph = tf.placeholder(tf.int64, [c.batch_size, c.time_steps],
                                         name='targets_ph')
        self.seq_length = tf.count_nonzero(self.inputs_ph, axis=1)

        self.cell = self.get_cell(c.layers, c)
        self.initial_state = self.cell.zero_state(c.batch_size, tf.float64)

        embedding = tf.get_variable("embedding", [c.vocab_size, c.cell_size], dtype=tf.float64)
        inputs = tf.nn.embedding_lookup(embedding, self.inputs_ph)

        outputs, self.final_state = tf.nn.dynamic_rnn(
            self.cell, inputs, initial_state=self.initial_state,
            sequence_length=self.seq_length, dtype=tf.float64,
        )
        output = tf.reshape(tf.concat(outputs, axis=1), [-1, c.cell_size])

        self.logits = tf.layers.dense(output, c.vocab_size, name='logits')
        self.probs = tf.nn.softmax(self.logits, name='probs')

    def get_cell(self, layers: int, c: ModelConfig):
        if layers > 1:
            return rnn.MultiRNNCell([
                self.get_cell(1, c)
                for _ in range(layers)
            ])
        cell = rnn.BasicLSTMCell(c.cell_size, c.forget_bias)
        assert 0 < c.input_keep_prob <= 1 and 0 < c.output_keep_prob <= 1
        if self.training and (0 < c.input_keep_prob < 1 or 0 < c.output_keep_prob < 1):
            cell = rnn.DropoutWrapper(cell, c.input_keep_prob, c.output_keep_prob)
        return cell
