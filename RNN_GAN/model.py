# Copyright 2017 Nan Du and CSE 847 group mates. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
import numpy as np

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_layers", 4,
                   "size of RNN hidden state")
flags.DEFINE_integer("rnn_size", 256,
                   "size of RNN hidden state")
flags.DEFINE_integer("batch_size", 64,
                   "minibatch size")
flags.DEFINE_integer("input_noise_size", 64,
                   "the noise size for generator")
flags.DEFINE_integer("vocab_size", 5000,
                   "size of vocabulary")
flags.DEFINE_integer("batch_size", 64,
                   "minibatch size")
flags.DEFINE_float("dropout", 0.5,
                   "rate of dropout")

FLAGS = flags.FLAGS

class GAN():
    def __init__(self, is_train = True):

        cell = rnn.BasicLSTMCell(FLAGS.rnn_size, state_is_tuple=False)
        if is_train:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=FLAGS.dropout)
        self.cell = cell = rnn.MultiRNNCell([cell] * FLAGS.num_layers, state_is_tuple=False)

        # input size is unknown
        self.input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

        self.targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
        self.initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [FLAGS.vocab_size, FLAGS.rnn_size])
            # in lstm-gnn inputs = input senstence
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        input_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_noise_size])
        input_noise_one_sent = tf.placeholder(tf.float32, [1, FLAGS.input_noise_size])
        self.input_noise = input_noise
        self.input_noise_one_sent = input_noise_one_sent



