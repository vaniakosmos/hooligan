import glob
import json

import numpy as np
import tensorflow as tf


def load_vocab(vocab_path: str):
    with open(vocab_path) as f:
        return json.load(f)


def decode_record(example_proto):
    features = {
        'tensor': tf.FixedLenFeature((), tf.string),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    tensor = tf.decode_raw(parsed_features['tensor'], tf.int64)  # type: tf.Tensor
    return tensor


class Dataset(object):
    def __init__(self, record_files, batch_size):
        if isinstance(record_files, str):
            record_files = glob.glob(record_files)
        ds = tf.data.TFRecordDataset(record_files)
        ds = ds.map(decode_record)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        ds = ds.repeat()
        iterator = ds.make_one_shot_iterator()
        self.get_next_op = iterator.get_next()

    def get_batch(self, sess: tf.Session):
        x = sess.run(self.get_next_op)
        y = np.copy(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = x[:, 0]
        return x, y


def test_next_op():
    ds = Dataset('../data/tweets.tfrecord', 10)
    np.set_printoptions(linewidth=2000)
    with tf.Session() as sess:
        x, y = ds.get_batch(sess)
        print(x[:30, :100])
        print(x.shape)


if __name__ == '__main__':
    test_next_op()
