# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.platform import gfile

import data_utils
from data_utils import IBO2_LABEL

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
        "model", "large",
        "A type of model. Possible options are: large.")
flags.DEFINE_string("data_dir", '/data/wordseg', "data_dir")
flags.DEFINE_string("train_dir", '/data/wordseg/training', "train_dir")
flags.DEFINE_integer("dump_per_steps", 200, "dump model after num of steps")
flags.DEFINE_boolean("do_test", 0, "do test only")

FLAGS = flags.FLAGS


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.is_training = is_training
        size = config.hidden_size
        vocab_size = config.vocab_size
        label_size = config.label_size
        self.global_step = tf.Variable(0, trainable=False)

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, label_size])
        softmax_b = tf.get_variable("softmax_b", [label_size])
        self.logits = logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self._targets, [-1])],
                [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.all_variables())

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1000
    max_epoch = 14
    max_max_epoch = 300
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 64
    vocab_size = 10000
    label_size = len(IBO2_LABEL)


class TestConfig(object):
    """for testing."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1000
    max_epoch = 14
    max_max_epoch = 300
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 1
    vocab_size = 10000
    label_size = len(IBO2_LABEL)


def run_epoch(session, m, data, eval_op, verbose=False, rev_vocab=None):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    state = m.initial_state.eval()
    iters = 0
    of = open('validate', 'w') if (not m.is_training and FLAGS.do_test) else None
    for step, (x, y) in enumerate(data_utils.data_iterator(data, m.batch_size,
                                                      m.num_steps)):
        cost, state, _, logits = session.run([m.cost, m.final_state, eval_op, m.logits],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps
        if verbose and (step + 1) % FLAGS.dump_per_steps == 0:
            print("current step: %s, perplexity: %.3f speed: %.0f wps" %
                  (m.global_step.eval(), np.exp(costs / iters),
                   FLAGS.dump_per_steps * m.batch_size / (time.time() - start_time)))

            if m.is_training:
                print ('dumping model...')
                m.saver.save(session, os.path.join(FLAGS.train_dir, "word_seg.ckpt"), global_step=m.global_step)
                print ('dumping model finished.')

        # testing
        if not m.is_training and FLAGS.do_test:
            assert of is not None
            labels = np.argmax(logits, axis=1)
            assert len(labels) == len(y[0])
            for i in xrange(len(labels)):
                if x[0][i] != data_utils.PAD_TOKEN_ID:
                    of.write('%s\t%s\t%s\n' % (x[0][i] if rev_vocab is None else rev_vocab[x[0][i]].encode('utf8'), y[0][i], labels[i]))
            of.write('\n')
    if of is not None:
        of.flush()
        of.close()

    return np.exp(costs / iters)


def get_config():
    if not FLAGS.do_test:
        return LargeConfig()
    else:
        return TestConfig()

def do_evaluation(path_to_eval_file):
    """
    evaluate word segment quality, per token error.
    :param path_to_eval_file:
    :return:
    """
    tp = 0
    fp = 0
    expected_token_count = 0
    actual_token_count = 0
    expect_words = []
    actual_words = []
    expect_word_frag = ''
    actual_word_frag = ''
    sentence_count = 0
    with open(path_to_eval_file) as evf:
        for line in evf:
            line = line.strip()
            # at the end of sentence
            if len(line) == 0:
                sentence_count += 1
                if len(expect_word_frag) > 0:
                    expect_words.append(expect_word_frag)
                    expect_word_frag = ''
                if len(actual_word_frag) > 0:
                    actual_words.append(actual_word_frag)
                    actual_word_frag = ''
                if len(expect_words) == 0:
                    sys.stderr.write('found tow empty line in a raw.')
                expected_token_count += len(expect_words)
                actual_token_count += len(actual_words)
                exs = ''
                acs = ''
                while len(expect_words) > 0:
                    if len(exs) == len(acs):
                        aw = actual_words.pop(0)
                        ew = expect_words.pop(0)
                        acs += aw
                        exs += ew
                        if aw == ew:
                            tp += 1
                        else:
                            fp += 1
                    while len(exs) != len(acs):
                        if len(exs) > len(acs):
                            aw = actual_words.pop(0)
                            acs += aw
                            fp += 1
                        else:
                            ew = expect_words.pop(0)
                            exs += ew
                assert len(expect_words) == len(actual_words) == 0
            else:
                char, expect_tag, actual_tag = line.split('\t')
                if expect_tag == '0' or expect_tag == '2':
                    if len(expect_word_frag) > 0:
                        expect_words.append(expect_word_frag)
                    expect_word_frag = char
                else:
                    expect_word_frag += char

                if actual_tag == '0' or actual_tag == '2':
                    if len(actual_word_frag) > 0:
                        actual_words.append(actual_word_frag)
                    actual_word_frag = char
                else:
                    actual_word_frag += char
    return tp, fp, actual_token_count, expected_token_count

def main(_):
    if not FLAGS.data_dir or not FLAGS.train_dir:
        raise ValueError("Must set --data_dir and --train_dir to data and training directory")

    vocab_path = data_utils.create_vocabulary(os.path.join(FLAGS.data_dir, 'train'), FLAGS.data_dir)
    train_data = data_utils.read_data(os.path.join(FLAGS.data_dir, 'train'), vocab_path)
    valid_data = data_utils.read_data(os.path.join(FLAGS.data_dir, 'dev'), vocab_path)
    test_data = valid_data

    _, rev_vocab = data_utils.read_vocabulary(vocab_path)


    config = get_config()

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=config)

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            tf.initialize_all_variables().run()

        if not FLAGS.do_test:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        else:
            print ("do testing on test data...")
            test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(), rev_vocab=rev_vocab)
            print("Test Perplexity: %.3f" % test_perplexity)
            tp, fp, actual_token_count, expected_token_count = do_evaluation('validate')
            print('precision: ', (tp + 0.0)/actual_token_count, ', recall: ', (tp + 0.0)/expected_token_count)


if __name__ == "__main__":
    tf.app.run()