import glob
import json

import numpy as np
import tensorflow as tf
from tensorflow.core.example.feature_pb2 import BytesList, Feature, Features

from rnn.model import ModelConfig


def iterate_txt(data_file: str):
    with open(data_file) as file:
        for line in file:
            line = line.strip()
            if line:
                yield line


def iterate_csv(data_file: str, column: str):
    pass


def encode_record(data):
    features = {
        'tensor': Feature(bytes_list=BytesList(value=[data.tobytes()])),
    }
    features = Features(feature=features)
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def decode_record(example_proto):
    features = {
        'tensor': tf.FixedLenFeature((), tf.string),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    tensor = tf.decode_raw(parsed_features['tensor'], tf.int64)  # type: tf.Tensor
    return tensor


def save_vocab(chars, vocab_path: str):
    chars = list(chars)
    chars = ['pad', 'break'] + chars  # 0 - pad, 1 - break
    vocab = dict(zip(chars, range(len(chars))))
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    return vocab


def load_vocab(vocab_path: str):
    with open(vocab_path) as f:
        return json.load(f)


class BuildConfig(ModelConfig):
    limit = 280
    data_file = '../data/input.txt'
    vocab_file = '../data/vocab.json'
    tfrecord_file = '../data/test.tfrecord'
    with_break = True


def prebuild():
    c = BuildConfig().define()
    writer = tf.python_io.TFRecordWriter(c.tfrecord_file)

    chars = set()
    for line in iterate_txt(c.data_file):
        chars = chars | set(line)
    vocab = save_vocab(chars, c.vocab_file)

    for line in iterate_txt(c.data_file):
        ids = list(map(vocab.get, line))
        if c.with_break:
            ids += [1]
        tensor = np.array(ids, dtype=np.int64)
        tensor = tensor[:c.limit]
        data = np.zeros([c.limit], dtype=np.int64)
        data[:len(tensor)] = tensor
        ex = encode_record(data)
        writer.write(ex)


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
    ds = Dataset('../data/test.tfrecord', 10)
    with tf.Session() as sess:
        x, y = ds.get_batch(sess)
        print(x[:, :10])
        print(x.shape)


if __name__ == '__main__':
    prebuild()
    # test_next_op()
