#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Author  :   Zhuang Di ZHU
#   E-mail  :   zhuangdizhu@yahoo.com
#   Date    :   17/03/18 18:35:56
#   Desc    :
#


import io
import sys
import collections
import numpy as np
import tensorflow as tf
import myRNN            # Added by Judy

# ----------------------------Constants---------------------------------#
reload(sys)
sys.setdefaultencoding('utf8')
poetry_file = 'poetry.txt'

batch_size = 64
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])

# ----------------------------数据预处理-------------------------------#
# 诗集
poetries = []
removed_poetries = []
with io.open(poetry_file, "r", encoding='utf-8') as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if not myRNN.ifChinese(content):
                continue
            if len(content) < 5 or len(content) > 80:
                removed_poetries.append(content)
                continue
            content = '[' + content + ']'  # ???? --By Judy
            poetries.append(content)
        except Exception as e:
            removed_poetries.append(content)  # Added by Judy
            pass

# 按诗的字数排序
print('唐诗总数: ' + str(len(poetries)))
print('未使用总数: ' + str(len(removed_poetries)))

poetries = sorted(poetries, key=lambda line: len(line))

# 统计每个字出现次数
# all_words = list(set(poetries))
all_words = []
for poetry in poetries:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)

# counter_pairs 结构： [(汉字,出现次数),(汉字,出现次数),...] 按照出现次数从大到小排序。--By Judy
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
# print(count_pairs[0][0]) 出现最多的是',' --By Judy

# words 结构:[汉字，汉字, 汉字,...] 按照出现次数从大到小排序. --By Judy
words, _ = zip(*count_pairs)

# 取前多少个常用字
count = len(words)
words = words[:count] + (' ',)

# 每个字映射为一个数字ID
# word_num_map结构: {汉字:0,汉字:,1,...} 出现次数最大的汉字对应数字是0.  --By Judy
word_num_map = dict(zip(words, range(len(words))))


# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))

# poetries_vector格式: size与poetries一致,汉字全部替换成数字。 --By Judy
poetries_vector = [list(map(to_num, poetry)) for poetry in poetries]

# 每次取64首诗进行训练
batch_size = 64
n_chunk = len(poetries_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size
    # batches 结构: a 64-length list, each element in the list is a poetry. --By Judy
    batches = poetries_vector[start_index:end_index]
    # max_length: the longest length of the above 64 poetries. --By Judy
    max_length = max(map(len, batches))
    # xdata is a matrix with size of 64 X max_length. --By Judy
    xdata = np.full((batch_size, max_length), word_num_map[' '], np.int32)

    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        ydata[:, :-1] = xdata[:, 1:]
        """
            xdata             ydata
            [6,2,4,6,9]       [2,4,6,9,9]
            [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(xdata)
        y_batches.append(ydata)

def train_neural_network(iteration):
    logits, last_state, _, _, _ = myRNN.neural_network(words, input_data, my_batch_size=batch_size)
    targets = tf.reshape(output_targets, [-1])
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],
                                                  len(words))
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        #saver = tf.train.Saver()

        for epoch in range(iteration):
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            n = 0
            for batche in range(n_chunk):
                train_loss, _, _ = sess.run([cost, last_state, train_op],
                                            feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
                n += 1
                print("epoch-{}  batch-{}  train loss-{}".format(epoch, batche, train_loss))
            if epoch % 7 == 0:
                saver.save(sess, 'poetry.module', global_step=epoch)



train_neural_network(5)