import numpy as np
import tensorflow as tf

from dataset import load_vocab
from model import Model, ModelConfig


class Config(ModelConfig):
    vocab_path = '../data/vocab.json'
    weights_path = '../weights'


def get_next(sess, model, state, value):
    x = np.zeros((1, 1))
    x[0, 0] = value
    feed = {
        model.inputs_ph: x,
        model.initial_state: state,
        model.seq_length: [1],
    }
    probs, state = sess.run((model.probs, model.final_state), feed)
    return probs[0], state


def sample(sess, model: Model, vocab: dict, num=200, prime='the'):
    assert len(prime) > 0
    result = []
    state = sess.run(model.cell.zero_state(1, tf.float64))
    char_id = None
    for char in prime:
        char_id = vocab[char]
        probs, state = get_next(sess, model, state, char_id)
        result.append(char_id)
    for _ in range(num):
        probs, state = get_next(sess, model, state, char_id)
        char_id = np.argmax(probs)
        result.append(char_id)
    return result


def decode(ids, vocab: dict):
    chars = {
        i: c for c, i in vocab.items()
    }
    return ''.join(map(chars.get, ids))


def main():
    c = Config().define()
    vocab = load_vocab(c.vocab_path)

    model = Model(c, training=True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(c.weights_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        res = sample(sess, model, vocab, num=100, prime='hello gamlet')
        res = decode(res, vocab)
        print(repr(res))


if __name__ == '__main__':
    main()
