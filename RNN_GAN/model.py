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
import time, datetime, os, sys
import tensorflow as tf

"""

The hyperparameters used in the model:
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- songlength - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- epochs_before_decay - the number of epochs trained with the initial learning rate
- max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "epochs_before_decay"
- batch_size - the batch size
"""

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 20,  # 10, 20
                     "Batch size.")

flags.DEFINE_float("reg_scale", 1.0,  #
                   "L2 regularization scale.")

flags.DEFINE_integer("hidden_size_g", 350,  # 200, 1500
                     "Hidden size for recurrent part of G.")
flags.DEFINE_integer("hidden_size_d", 350,  # 200, 1500
                     "Hidden size for recurrent part of D. Default: same as for G.")
flags.DEFINE_float("keep_prob", 0.5,  # 1.0, .35
                   "Keep probability. 1.0 disables dropout.")

FLAGS = flags.FLAGS


def data_type():
    # return tf.float16 if FLAGS.float16 else tf.float32
    return tf.float32

def linear(inp, output_dim, scope=None, stddev=1.0, reuse_scope=False):
    norm = tf.random_normal_initializer(stddev=stddev, dtype=data_type())
    const = tf.constant_initializer(0.0, dtype=data_type())
    with tf.variable_scope(scope or 'linear') as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
        if reuse_scope:
            scope.reuse_variables()
        # print('inp.get_shape(): {}'.format(inp.get_shape()))
        w = tf.get_variable('w', [inp.get_shape()[1], output_dim], initializer=norm, dtype=data_type())
        b = tf.get_variable('b', [output_dim], initializer=const, dtype=data_type())
    return tf.matmul(inp, w) + b

class RNNGAN(object):
    """The RNNGAN model."""

    def __init__(self, is_training, num_song_features=None, num_meta_features=None):
        self.batch_size = batch_size = FLAGS.batch_size
        #self.songlength = songlength = FLAGS.songlength
        # self.global_step            = tf.Variable(0, trainable=False)

        #print('songlength: {}'.format(self.songlength))
        #self._input_songdata = tf.placeholder(shape=[batch_size, songlength, num_song_features], dtype=data_type())
        #self._input_metadata = tf.placeholder(shape=[batch_size, num_meta_features], dtype=data_type())
        self._input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

        songdata_inputs = [tf.squeeze(input_, [1])
                           for input_ in tf.split(1, None, self._input_data)]

        with tf.variable_scope('G') as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size_g, forget_bias=1.0, state_is_tuple=True)
            if is_training and FLAGS.keep_prob < 1:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=FLAGS.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.num_layers_g, state_is_tuple=True)

            self._initial_state = cell.zero_state(batch_size, tf.float32)

            # TODO: (possibly temporarily) disabling meta info
            """
            if FLAGS.generate_meta:
                metainputs = tf.random_uniform(shape=[batch_size, int(FLAGS.random_input_scale * num_meta_features)],
                                               minval=0.0, maxval=1.0)
                meta_g = tf.nn.relu(linear(metainputs, FLAGS.meta_layer_size, scope='meta_layer', reuse_scope=False))
                meta_softmax_w = tf.get_variable("meta_softmax_w", [FLAGS.meta_layer_size, num_meta_features])
                meta_softmax_b = tf.get_variable("meta_softmax_b", [num_meta_features])
                meta_logits = tf.nn.xw_plus_b(meta_g, meta_softmax_w, meta_softmax_b)
                meta_probs = tf.nn.softmax(meta_logits)
            """
            random_rnninputs = tf.random_uniform(
                shape=[batch_size, None, int(FLAGS.random_input_scale * num_song_features)], minval=0.0,
                maxval=1.0, dtype=data_type())
            
            # Make list of tensors. One per step in recurrence.
            # Each tensor is batchsize*numfeatures.
            random_rnninputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, None, random_rnninputs)]

            # REAL GENERATOR:
            state = self._initial_state
            # as we feed the output as the input to the next, we 'invent' the initial 'output'.
            generated_point = tf.random_uniform(shape=[batch_size, num_song_features], minval=0.0, maxval=1.0,
                                                dtype=data_type())
            outputs = []
            self._generated_features = []
            for i, input_ in enumerate(random_rnninputs):
                if i > 0: scope.reuse_variables()
                concat_values = [input_]
                if not FLAGS.disable_feed_previous:
                    concat_values.append(generated_point)
                """
                if FLAGS.generate_meta:
                    concat_values.append(meta_probs)
                """

                if len(concat_values):
                    input_ = tf.concat(concat_dim=1, values=concat_values)
                input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_g,
                                           scope='input_layer', reuse_scope=(i != 0)))
                output, state = cell(input_, state)
                outputs.append(output)
                # generated_point = tf.nn.relu(linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0)))
                generated_point = linear(output, num_song_features, scope='output_layer', reuse_scope=(i != 0))
                self._generated_features.append(generated_point)

            # PRETRAINING GENERATOR, will feed inputs, not generated outputs:
            scope.reuse_variables()
            # as we feed the output as the input to the next, we 'invent' the initial 'output'.
            prev_target = tf.random_uniform(shape=[batch_size, num_song_features], minval=0.0, maxval=1.0,
                                            dtype=data_type())
            outputs = []
            self._generated_features_pretraining = []
            for i, input_ in enumerate(random_rnninputs):
                concat_values = [input_]
                if not FLAGS.disable_feed_previous:
                    concat_values.append(prev_target)

                """
                if FLAGS.generate_meta:
                    concat_values.append(self._input_metadata)
                """
                if len(concat_values):
                    input_ = tf.concat(concat_dim=1, values=concat_values)
                input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_g, scope='input_layer', reuse_scope=(i != 0)))
                output, state = cell(input_, state)
                outputs.append(output)
                # generated_point = tf.nn.relu(linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0)))
                generated_point = linear(output, num_song_features, scope='output_layer', reuse_scope=(i != 0))
                self._generated_features_pretraining.append(generated_point)
                prev_target = songdata_inputs[i]

                # outputs, state = tf.nn.rnn(cell, transformed, initial_state=self._initial_state)

                # self._generated_features = [tf.nn.relu(linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0))) for i,output in enumerate(outputs)]

        self._final_state = state

        # These are used both for pretraining and for D/G training further down.
        self._lr = tf.Variable(FLAGS.learning_rate, trainable=False, dtype=data_type())
        self.g_params = [v for v in tf.trainable_variables() if v.name.startswith('model/G/')]
        if FLAGS.adam:
            g_optimizer = tf.train.AdamOptimizer(self._lr)
        else:
            g_optimizer = tf.train.GradientDescentOptimizer(self._lr)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.1  # Choose an appropriate one.
        reg_loss = reg_constant * sum(reg_losses)
        reg_loss = tf.Print(reg_loss, reg_losses,
                            'reg_losses = ', summarize=20, first_n=20)
        # if not FLAGS.disable_l2_regularizer:
        #  print('L2 regularization. Reg losses: {}'.format([v.name for v in reg_losses]))

        # ---BEGIN, PRETRAINING. ---

        print(tf.transpose(tf.pack(self._generated_features_pretraining), perm=[1, 0, 2]).get_shape())
        print(self._input_data.get_shape())
        self.rnn_pretraining_loss = tf.reduce_mean(
            tf.squared_difference(x=tf.transpose(tf.pack(self._generated_features_pretraining), perm=[1, 0, 2]),
                                  y=self._input_songdata))
        if not FLAGS.disable_l2_regularizer:
            self.rnn_pretraining_loss = self.rnn_pretraining_loss + reg_loss

        pretraining_grads, _ = tf.clip_by_global_norm(tf.gradients(self.rnn_pretraining_loss, self.g_params),
                                                      FLAGS.max_grad_norm)
        self.opt_pretraining = g_optimizer.apply_gradients(zip(pretraining_grads, self.g_params))

        # ---END, PRETRAINING---

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('D') as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
            # Make list of tensors. One per step in recurrence.
            # Each tensor is batchsize*numfeatures.
            # TODO: (possibly temporarily) disabling meta info
            print('self._input_songdata shape {}'.format(self._input_songdata.get_shape()))
            print('generated data shape {}'.format(self._generated_features[0].get_shape()))
            # TODO: (possibly temporarily) disabling meta info
            if FLAGS.generate_meta:
                songdata_inputs = [tf.concat(1, [self._input_metadata, songdata_input]) for songdata_input in
                                   songdata_inputs]
            # print('metadata inputs shape {}'.format(self._input_metadata.get_shape()))
            # print('generated metadata shape {}'.format(meta_probs.get_shape()))
            self.real_d, self.real_d_features = self.discriminator(songdata_inputs, is_training, msg='real')
            scope.reuse_variables()
            # TODO: (possibly temporarily) disabling meta info
            if FLAGS.generate_meta:
                generated_data = [tf.concat(1, [meta_probs, songdata_input]) for songdata_input in
                                  self._generated_features]
            else:
                generated_data = self._generated_features
            if songdata_inputs[0].get_shape() != generated_data[0].get_shape():
                print('songdata_inputs shape {} != generated data shape {}'.format(songdata_inputs[0].get_shape(),
                                                                                   generated_data[0].get_shape()))
            self.generated_d, self.generated_d_features = self.discriminator(generated_data, is_training,
                                                                             msg='generated')

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.d_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.real_d, 1e-1000000, 1.0)) \
                                     - tf.log(1 - tf.clip_by_value(self.generated_d, 0.0, 1.0 - 1e-1000000)))
        self.g_loss_feature_matching = tf.reduce_sum(
            tf.squared_difference(self.real_d_features, self.generated_d_features))
        self.g_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.generated_d, 1e-1000000, 1.0)))

        if not FLAGS.disable_l2_regularizer:
            self.d_loss = self.d_loss + reg_loss
            self.g_loss_feature_matching = self.g_loss_feature_matching + reg_loss
            self.g_loss = self.g_loss + reg_loss
        self.d_params = [v for v in tf.trainable_variables() if v.name.startswith('model/D/')]

        if not is_training:
            return

        d_optimizer = tf.train.GradientDescentOptimizer(self._lr * FLAGS.d_lr_factor)
        d_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_params),
                                            FLAGS.max_grad_norm)
        self.opt_d = d_optimizer.apply_gradients(zip(d_grads, self.d_params))
        if FLAGS.feature_matching:
            g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss_feature_matching,
                                                             self.g_params),
                                                FLAGS.max_grad_norm)
        else:
            g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params),
                                                FLAGS.max_grad_norm)
        self.opt_g = g_optimizer.apply_gradients(zip(g_grads, self.g_params))

        self._new_lr = tf.placeholder(shape=[], name="new_learning_rate", dtype=data_type())
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def discriminator(self, inputs, is_training, msg=''):
        # RNN discriminator:
        # for i in xrange(len(inputs)):
        #  print('shape inputs[{}] {}'.format(i, inputs[i].get_shape()))
        # inputs[0] = tf.Print(inputs[0], [inputs[0]],
        #        '{} inputs[0] = '.format(msg), summarize=20, first_n=20)
        if is_training and FLAGS.keep_prob < 1:
            inputs = [tf.nn.dropout(inp, FLAGS.keep_prob) for inp in inputs]
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size_d, forget_bias=1.0, state_is_tuple=True)
        if is_training and FLAGS.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=FLAGS.keep_prob)
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.num_layers_d, state_is_tuple=True)
        self._initial_state_fw = cell_fw.zero_state(self.batch_size, data_type())
        if not FLAGS.unidirectional_d:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size_g, forget_bias=1.0, state_is_tuple=True)
            if is_training and FLAGS.keep_prob < 1:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=FLAGS.keep_prob)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.num_layers_d, state_is_tuple=True)
            self._initial_state_bw = cell_bw.zero_state(self.batch_size, data_type())

            outputs, state_fw, state_bw = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                                  initial_state_fw=self._initial_state_fw,
                                                                  initial_state_bw=self._initial_state_bw)
            # outputs[0] = tf.Print(outputs[0], [outputs[0]],
            #        '{} outputs[0] = '.format(msg), summarize=20, first_n=20)
            # state = tf.concat(state_fw, state_bw)
            # endoutput = tf.concat(concat_dim=1, values=[outputs[0],outputs[-1]])
        else:
            outputs, state = tf.nn.rnn(cell_fw, inputs, initial_state=self._initial_state_fw)
            # endoutput = outputs[-1]

        if FLAGS.minibatch_d:
            outputs = [minibatch(tf.reshape(outp, shape=[FLAGS.batch_size, -1]), msg=msg, reuse_scope=(i != 0)) for
                       i, outp in enumerate(outputs)]
        # decision = tf.sigmoid(linear(outputs[-1], 1, 'decision'))
        if FLAGS.end_classification:
            decisions = [tf.sigmoid(linear(output, 1, 'decision', reuse_scope=(i != 0))) for i, output in
                         enumerate([outputs[0], outputs[-1]])]
            decisions = tf.pack(decisions)
            decisions = tf.transpose(decisions, perm=[1, 0, 2])
            print('shape, decisions: {}'.format(decisions.get_shape()))
        else:
            decisions = [tf.sigmoid(linear(output, 1, 'decision', reuse_scope=(i != 0))) for i, output in
                         enumerate(outputs)]
            decisions = tf.pack(decisions)
            decisions = tf.transpose(decisions, perm=[1, 0, 2])
            print('shape, decisions: {}'.format(decisions.get_shape()))
        decision = tf.reduce_mean(decisions, reduction_indices=[1, 2])
        decision = tf.Print(decision, [decision],
                            '{} decision = '.format(msg), summarize=20, first_n=20)
        return (decision, tf.transpose(tf.pack(outputs), perm=[1, 0, 2]))

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def generated_features(self):
        return self._generated_features

    @property
    def input_songdata(self):
        return self._input_songdata

    @property
    def input_metadata(self):
        return self._input_metadata

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
