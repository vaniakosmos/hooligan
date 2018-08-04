import json
import os

import numpy as np
import tensorflow as tf

from rnn.model import Model, ModelConfig


class NetConfig(ModelConfig):
    vocab_path = '../data/vocab.json'
    model_path = '../weights/l=2,cs=128'


class CharRNN(object):
    def __init__(self, c: NetConfig):
        if not c.vocab_path:
            c.vocab_path = os.path.join(os.path.dirname(c.model_path), 'vocab.json')

        with open(c.vocab_path) as f:
            self.char_to_id = json.load(f)  # type: dict
            self.id_to_char = {i: c for c, i in self.char_to_id.items()}

        self.model = Model(c, training=False)
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.load_graph(c.model_path)

    def load_graph(self, model_path: str):
        saver = tf.train.Saver(tf.global_variables())

        if os.path.isdir(model_path):
            ckpt = tf.train.get_checkpoint_state(model_path)
            model_path = ckpt and ckpt.model_checkpoint_path
        if model_path:
            saver.restore(self.sess, model_path)
        else:
            raise FileNotFoundError("Failed to find model file.")

    def get_next(self, state, char_id):
        feed_dict = {
            self.model.inputs_ph: [[char_id]],
            self.model.seq_length: [1],
        }
        if state:
            feed_dict[self.model.initial_state] = state
        probs, state = self.sess.run((
            self.model.probs, self.model.final_state,
        ), feed_dict)
        return probs[0], state

    def sample(self, start: str, size=280):
        assert len(start) > 0
        ids = []
        state = None
        char_id = None
        for char in start:
            char_id = self.char_to_id[char]
            probs, state = self.get_next(state, char_id)
            ids.append(char_id)
        for _ in range(size):
            probs, state = self.get_next(state, char_id)
            char_id = np.argmax(probs)
            ids.append(char_id)
        return self.decode_ids(ids)

    def decode_ids(self, ids):
        return ''.join(map(self.id_to_char.get, ids))

    def close(self):
        self.sess.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()


def main():
    c = NetConfig().define()
    net = CharRNN(c)
    print('>', net.sample('the'))
    print('>', net.sample('hello'))
    print('>', net.sample('foo'))
    print('>', net.sample('bar'))


if __name__ == '__main__':
    main()
