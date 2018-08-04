import html
import itertools
import json
import easy_flags

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.core.example.feature_pb2 import BytesList, Feature, Features


class BuildConfig(easy_flags.BaseConfig):
    time_steps = 281
    data_file = '../data/tweets.csv'
    vocab_path = '../data/vocab.json'
    tfrecord_file = '../data/tweets.tfrecord'
    with_break = True


def iterate_txt(data_file: str):
    with open(data_file) as file:
        for line in file:
            line = line.strip()
            if line:
                yield line


def iterate_troll_csv(data_file: str):
    df = pd.read_csv(data_file)
    df = df.loc[(df.language == 'English') & df.content.notnull()]
    for line in df['content']:
        try:
            line.encode('ascii')
        except Exception:
            continue
        line = html.unescape(line)
        words = line.split()
        words = list(itertools.dropwhile(lambda w: w.startswith('@'), words))
        if words[-1].startswith('https://t.co'):
            words.pop()
        filtered_line = ' '.join(words)
        yield filtered_line


def encode_record(data):
    features = {
        'tensor': Feature(bytes_list=BytesList(value=[data.tobytes()])),
    }
    features = Features(feature=features)
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def save_vocab(chars, vocab_path: str):
    chars = list(chars)
    chars = ['pad', 'break'] + chars  # 0 - pad, 1 - break
    vocab = dict(zip(chars, range(len(chars))))
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, ensure_ascii=False)
    return vocab


def get_iterator(data_file: str):
    if data_file.endswith('.csv'):
        return iterate_troll_csv(data_file)
    return iterate_txt(data_file)


def prebuild():
    c = BuildConfig().define().print()
    writer = tf.python_io.TFRecordWriter(c.tfrecord_file)

    chars = set()
    for i, line in enumerate(get_iterator(c.data_file)):
        if i % 10000 == 0:
            print(i)
        chars |= set(line)
    vocab = save_vocab(chars, c.vocab_path)
    print('vocab saved')

    for i, line in enumerate(get_iterator(c.data_file)):
        if i % 10000 == 0:
            print(i)
        ids = list(map(vocab.get, line))
        if c.with_break:
            ids += [1]
        tensor = np.array(ids, dtype=np.int64)
        tensor = tensor[:c.time_steps]
        data = np.zeros([c.time_steps], dtype=np.int64)
        data[:len(tensor)] = tensor
        ex = encode_record(data)
        writer.write(ex)


if __name__ == '__main__':
    prebuild()
