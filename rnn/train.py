import os
import time

import tensorflow as tf

from rnn.dataset import Dataset, load_vocab
from rnn.model import Model, ModelConfig


class TrainConfig(ModelConfig):
    epochs = 100
    lr = 0.01
    decay_rate = 0.97
    log_dir = '../logs'
    save_dir = '../weights'
    ds_path = '../data/test.tfrecord'
    log_step = 1
    save_step = 100


def get_model_name(c: TrainConfig):
    return f"l={c.layers},cs={c.cell_size}"


def restore_model(sess: tf.Session, saver: tf.train.Saver, c: TrainConfig):
    model_name = get_model_name(c)
    save_dir = os.path.join(c.save_dir, model_name)
    model_path = os.path.join(c.save_dir, model_name, 'model')
    latest_ckpt = tf.train.latest_checkpoint(save_dir)
    if latest_ckpt:
        print(f'💽 restoring latest checkpoint from: {latest_ckpt}')
        saver.restore(sess, latest_ckpt)
    else:
        print(f'💽 training new model')
    return model_path


def train():
    c = TrainConfig().define().print()
    vocab = load_vocab(c.vocab_path)
    c.vocab_size = len(vocab)

    model = Model(c)

    weights = tf.reshape(
        tf.sequence_mask(model.seq_length, maxlen=c.time_steps, dtype=tf.float64),
        shape=[c.batch_size * c.time_steps]
    )
    fat_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        logits=[model.logits],
        targets=[tf.reshape(model.targets_ph, [-1])],
        weights=[weights]
    )
    loss = tf.reduce_sum(fat_loss) / c.batch_size

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), c.grad_clip)
    lr = tf.Variable(c.lr, trainable=False)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    ds = Dataset(c.ds_path, c.batch_size)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        save_path = restore_model(sess, saver, c)

        # summary stuff
        tf.summary.histogram('logits', model.logits)
        tf.summary.histogram('loss', fat_loss)
        tf.summary.scalar('train_loss', loss)
        summary_op = tf.summary.merge_all()
        summary_dir = os.path.join(c.log_dir, get_model_name(c), time.strftime('%Y.%m.%d:%H.%M.%S'))
        summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=sess.graph)

        # train
        for _ in range(c.epochs):
            x, y = ds.get_batch(sess)
            loss_, step, summary, _ = sess.run((loss, global_step, summary_op, train_op), {
                model.inputs_ph: x,
                model.targets_ph: y,
            })

            if step % c.log_step == 0:
                print(f"🔊 {step:-6d} - loss={loss_:.5f}")
                summary_writer.add_summary(summary, step)

            if step > 0 and step % c.save_step == 0:
                saved_path = saver.save(sess, save_path, global_step=step)
                print(f"💾 model saved to {saved_path}")


if __name__ == '__main__':
    train()
