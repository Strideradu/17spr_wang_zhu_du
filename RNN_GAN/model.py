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
flags.DEFINE_integer("hide_size_G", 512,
                   "size of generator hidden state")
flags.DEFINE_integer("hide_size_D", 512,
                   "size of discriminator hidden state")
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
    def __init__(self, first_input, is_train = True):
        self.first_input = first_input

        cell = rnn.BasicLSTMCell(FLAGS.rnn_size, state_is_tuple=False)
        if is_train:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=FLAGS.dropout)
        self.cell = cell = rnn.MultiRNNCell([cell] * FLAGS.num_layers, state_is_tuple=False)

        # input size is unknown
        self.input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

        self.targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
        self.initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

        #with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [FLAGS.vocab_size, FLAGS.rnn_size])
        self.embedding = embedding
        # in lstm-gnn inputs = input senstence
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        input_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_noise_size])
        input_noise_one_sent = tf.placeholder(tf.float32, [1, FLAGS.input_noise_size])
        self.input_noise = input_noise
        self.input_noise_one_sent = input_noise_one_sent

        _, gen_vars = self.build_generator(input_noise, is_train=True)

    def build_generator(self, input_, reuse=False, is_train=False):
        embedding, first_input = self.embedding, self.first_input

        with tf.variable_scope('generator_model', reuse=reuse):
            input_noise_w = tf.get_variable(
                "input_noise_w",
                [FLAGS.input_noise_size, FLAGS.hide_size_G],
                initializer=tf.random_normal_initializer(0, stddev=1 / np.sqrt(FLAGS.vocab_size))
            )
            input_noise_b = tf.get_variable(
                "input_noise_b",
                [FLAGS.hide_size_G],
                initializer=tf.constant_initializer(1e-4)
            )

            first_hidden_state = tf.nn.relu(tf.matmul(input_, input_noise_w) + input_noise_b)

            cell = tf.nn.rnn_cell.GRUCell(FLAGS.hide_size_G)
            if is_train:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=FLAGS.dropout)

            input_w = tf.get_variable(
                "input_w",
                [FLAGS.vocab_size,FLAGS.hide_size_G],
                initializer=tf.random_normal_initializer(0, stddev=1 / np.sqrt(FLAGS.vocab_size))
            )
            input_b = tf.get_variable(
                "input_b",
                [FLAGS.hide_size_G],
                initializer=tf.constant_initializer(1e-4)
            )

            softmax_w = tf.get_variable(
                "softmax_w",
                [FLAGS.hide_size_G, FLAGS.vocab_size],
                initializer=tf.random_normal_initializer(0, stddev=1 / np.sqrt(FLAGS.hide_size_G))
            )
            softmax_b = tf.get_variable(
                "softmax_b",
                [FLAGS.vocab_size],
                initializer=tf.constant_initializer(1e-4)
            )

            state = first_hidden_state

            labels = tf.fill([tf.shape(input_)[0], 1], tf.cast(first_input, tf.int32))
            input_ = tf.nn.embedding_lookup(embedding, labels)

            outputs = []
            with tf.variable_scope("GRU_generator"):
                for time_step in range(seq_size):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    inp = tf.nn.relu(tf.matmul(input_[:, 0, :], input_w) + input_b)

                    cell_output, state = cell(inp, state)
                    logits = tf.nn.softmax(tf.matmul(cell_output, softmax_w) + softmax_b)
                    labels = tf.expand_dims(tf.argmax(logits, 1), 1)
                    input_ = tf.nn.embedding_lookup(embedding, labels)
                    outputs.append(tf.expand_dims(logits, 1))

            output = tf.concat(1, outputs)
        variables = [v for v in tf.all_variables() if 'generator_model' in v.name]

        return output, variables

    def build_discriminator(self, input_, is_train = False, reuse = False):


        with tf.variable_scope('discriminator_model', reuse = reuse):
            cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size_D)
            if is_train:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            if is_train:
                input_ = tf.nn.dropout(input_, FLAGS.dropout)

            state = cell.zero_state(FLAGS.batch_size, tf.float32)

            input_w = tf.get_variable(
                "input_w",
                [vocab_size, FLAGS.hidden_size_D],
                initializer=tf.random_normal_initializer(0, stddev=1/np.sqrt(vocab_size))
            )
            input_b = tf.get_variable(
                "input_b",
                [FLAGS.hidden_size_D],
                initializer=tf.constant_initializer(1e-4)
            )

            with tf.variable_scope("GRU_discriminator"):
                for time_step in range(seq_size):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    inp = tf.nn.relu(tf.matmul(input_[:, time_step, :], input_w) + input_b)
                    cell_output, state = cell(inp, state)

            out_w = tf.get_variable(
                "discriminator_output_w",
                [FLAGS.hidden_size_D, 1],
                initializer=tf.random_normal_initializer(0, 1./np.sqrt(FLAGS.hidden_size_D))
            )
            out_b = tf.get_variable(
                "discriminator_output_b",
                [1],
                initializer=tf.constant_initializer(1e-4)
            )

            output = tf.reduce_mean(tf.sigmoid(tf.matmul(cell_output, out_w) + out_b))

        variables = [v for v in tf.all_variables() if 'discriminator_model' in v.name]

        return output, variables
