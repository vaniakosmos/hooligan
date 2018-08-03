import easy_flags


class ModelConfig(easy_flags.BaseConfig):
    layers = 2
    cell_size = 128
    forget_bias = 0.8
    input_keep_prob = 0.9
    output_keep_prob = 0.9
    time_steps = 280  # tweet size
    vocab_size = 64
    grad_clip = 5
    batch_size = 64
    vocab_path = '../data/vocab.json'


class TrainConfig(ModelConfig):
    epochs = 20000
    lr = 0.01
    decay_rate = 0.97
    log_dir = '../logs'
    save_dir = '../weights'
    ds_path = '../data/test.tfrecord'
    log_step = 1
    save_step = 100


class BakeConfig(ModelConfig):
    batch_size = 1
    time_steps = 1
