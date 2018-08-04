import glob
import json
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.core.example.feature_pb2 import BytesList, Feature, Features, Int64List

from rnn.model import ModelConfig


def iterate_txt(data_file: str):
    with open(data_file) as file:
        for line in file:
            line = line.strip()
            if line:
                yield line


def iterate_csv(data_file: str, column: str):
    pass


def encode_record(data, length):
    features = {
        'tensor': Feature(bytes_list=BytesList(value=[data.tobytes()])),
        'length': Feature(int64_list=Int64List(value=[length])),
    }
    features = Features(feature=features)
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def decode_record(example_proto):
    features = {
        'tensor': tf.FixedLenFeature((), tf.string),
        'length': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    tensor = tf.decode_raw(parsed_features['tensor'], tf.int64)  # type: tf.Tensor
    length = parsed_features['length']
    return tensor, length


def save_vocab(counter: Counter, vocab_path: str):
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    return vocab


def load_vocab(vocab_path: str):
    with open(vocab_path) as f:
        return json.load(f)


def prebuild():
    c = ModelConfig().define()
    limit = c.time_steps
    data_file = '../data/input.txt'
    vocab_file = '../data/vocab.json'
    tfrecord_file = '../data/test.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecord_file)

    counter = Counter()
    for line in iterate_txt(data_file):
        counter.update(line)
    vocab = save_vocab(counter, vocab_file)

    for line in iterate_txt(data_file):
        tensor = np.array(list(map(vocab.get, line)), dtype=np.int64)
        tensor = tensor[:limit]
        data = np.zeros([limit], dtype=int)
        data[:len(tensor)] = tensor
        ex = encode_record(data, len(tensor))
        writer.write(ex)


class Dataset:
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
        x, sl = sess.run(self.get_next_op)
        y = np.copy(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = x[:, 0]
        return x, y, sl


def test_next_op():
    ds = Dataset('../data/test.tfrecord', 10)
    with tf.Session() as sess:
        x, _, y = ds.get_batch(sess)
        print(x)
        print(x.shape)
        print(y)
        print(y.shape)


if __name__ == '__main__':
    prebuild()
    # test_next_op()
