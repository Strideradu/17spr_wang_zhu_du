#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Author  :   Zhuang Di ZHU
#   E-mail  :   zhuangdizhu@yahoo.com
#   Date    :   17/03/19 13:13:33
#   Desc    :
#
import tensorflow as tf

def ifChinese(content):
    punctuations = ['\xef\xbc\x8c', '\xe3\x80\x82', ',', ' ', '', '\xef\xbc\x9f']  # ? 。, 空格
    for c in content:
        if not u'\u4e00' <= c <= u'\u9fff':
            if c not in punctuations:
                # print(c)
                return False
    return True

# 定义RNN
def neural_network(words, input_data, my_batch_size=64, model='lstm', rnn_size=128, num_layers=2):
    print('RNN begins ...')
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell  # --By Judy
        # cell_fun = tf.nn.rnn_cell.BasicLSTMCell
    else:
        pass

    print('Create cell ...')
    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    initial_state = cell.zero_state(my_batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words) + 1])
        softmax_b = tf.get_variable("softmax_b", [len(words) + 1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words) + 1, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])

    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return (logits, last_state, probs, cell, initial_state)  # 训练
