#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import numpy as np

class Model():
    def __init__(self, args,infer=False):
        self.args = args
        if infer:
            args.batch_size = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))


        cell = cell_fn(args.rnn_size,state_is_tuple=False)
        self.cell = cell = rnn.MultiRNNCell([cell] * args.num_layers,state_is_tuple=False)


        ######   Add dropout: uncoment this block of code when you need ######
        #
        #if not infer:
        #    # training case
        #    cell_dropout = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=dropout, output_keep_prob=dropout)
        #    self.cell = cell = rnn.MultiRNNCell([cell_dropout] * args.num_layers,state_is_tuple=False)
        #else:
        #    # testing case
        #    self.cell = cell = rnn.MultiRNNCell([cell] * args.num_layers,state_is_tuple=False)
        #
        ######   Add dropout: uncoment this block of code when you need ######


        # the length of input sequence is variable.

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, None])

        self.input_rhyme = tf.placeholder(tf.int32, [args.batch_size, None])
        #self.input_rhyme = tf.placeholder(tf.int32, [args.batch_size, args.poem_length])

        self.target_data = tf.placeholder(tf.int32, [args.batch_size, None])

        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                word_embedding = tf.get_variable("word_embedding", [args.vocab_size, args.rnn_size])
                input_data = tf.nn.embedding_lookup(word_embedding, self.input_data)

                #rhyme_embedding = tf.get_variable("rhyhme_embedding", [args.vocab_size, args.rnn_size])
                #inputs_rhyme = tf.nn.embedding_lookup(rhyme_embedding, self.input_rhyme)

                #total_inputs = tf.concat([inputs_data, inputs_rhyme], 2)

        outputs, last_state = tf.nn.dynamic_rnn(cell,
                                                input_data,
                                                initial_state=self.initial_state,
                                                scope='rnnlm')
        output = tf.reshape(outputs,[-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b

        self.probs = tf.nn.softmax(self.logits)
        target_data = tf.reshape(self.target_data, [-1])
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [target_data],
                [tf.ones_like(target_data,dtype=tf.float32)],
                args.vocab_size)
        self.cost = tf.reduce_mean(loss)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, rhymes, prime=u'', sampling_type=1, cipai_rules=None):

        def pick_char(weights):
            if sampling_type == 0:
                sample = np.argmax(weights)
            else:
                t = np.cumsum(weights)
                s = np.sum(weights)
                sample = int(np.searchsorted(t, np.random.rand(1)*s))
            return chars[sample]

        # prime is a list of chinese characters that you want it to show up in
        # the begining of every sentence in your poem
        for char in prime:
            if char not in vocab:
                return u"{} is not in charset!".format(char)

        if not prime:
            state = self.cell.zero_state(1, tf.float32).eval()
            prime = u'^'
            result = u''
            x = np.array([list(map(vocab.get,prime))])

            #xrhyme = np.array([list(map(rhymes.get,prime))])

            # generateing the first character
            [probs,state] = sess.run([self.probs,self.final_state],
                                     {self.input_data: x,
                                      self.initial_state: state})
            char = pick_char(probs[-1])

            while char != u'$':
                result += char
                x = np.zeros((1,1))
                x[0,0] = vocab[char]
                [probs,state] = sess.run([self.probs,
                                          self.final_state],
                                         {self.input_data: x,
                                          self.initial_state: state})
                char = pick_char(probs[-1])
            return result
        else:
            result = u'^'
            for prime_char in prime:
                result += prime_char
                x = np.array([list(map(vocab.get,result))])
                state = self.cell.zero_state(1, tf.float32).eval()
                [probs,state] = sess.run([self.probs,self.final_state],{self.input_data: x,self.initial_state: state})
                char = pick_char(probs[-1])
                while char != u'，' and char != u'。':
                    result += char
                    x = np.zeros((1,1))
                    x[0,0] = vocab[char]
                    [probs,state] = sess.run([self.probs,self.final_state],{self.input_data: x,self.initial_state: state})
                    char = pick_char(probs[-1])
                result += char
            return result[1:]
