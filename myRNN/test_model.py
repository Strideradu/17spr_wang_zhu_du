#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Author  :   Zhuang Di ZHU
#   E-mail  :   zhuangdizhu@yahoo.com
#   Date    :   17/03/18 18:36:31
#   Desc    :
#

import sys,io
import collections
import numpy as np
import tensorflow as tf
import myRNN  #Added by Judy

reload(sys)
sys.setdefaultencoding('utf8')
poetry_file = 'poetry.txt'
# -------------------------------数据预处理---------------------------#

# 诗集
poetry_file = 'poetry.txt'
poetries = []
removed_poetries = []
with io.open(poetry_file, "r", encoding='utf-8') as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if not myRNN.ifChinese(content):
                continue
            if len(content) < 5 or len(content) > 200:
                removed_poetries.append(content)
                continue
            content = '[' + content + ']'  # ???? --By Judy
            poetries.append(content)
        except Exception as e:
            removed_poetries.append(content)  # Added by Judy
            pass



# 按诗的字数排序
poetries = sorted(poetries, key=lambda line: len(line))
print('唐诗总数: ' + str(len(poetries)))
print('未使用总数: ' + str(len(removed_poetries)))

# 统计每个字出现次数
all_words = []
for poetry in poetries:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)

# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetries_vector = [list(map(to_num, poetry)) for poetry in poetries]

# [[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
# [339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
# ....]

batch_size = 1
n_chunk = len(poetries_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size

    batches = poetries_vector[start_index:end_index]
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
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

# ---------------------------------------RNN--------------------------------------#

input_data = tf.placeholder(tf.int32, [batch_size, None])




# -------------------------------生成古诗---------------------------------#
# 使用训练完成的模型
def to_word(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return words[sample]


def gen_poetry():
    _, last_state, probs, cell, initial_state = myRNN.neural_network(words, input_data, my_batch_size=batch_size)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        module_file = tf.train.latest_checkpoint('.')

        new_saver = tf.train.import_meta_graph(module_file+".meta")
        #new_saver = tf.train.Saver(tf.all_variables())
        new_saver.restore(sess, module_file)

        state_ = sess.run(cell.zero_state(1, tf.float32))

        x = np.array([list(map(word_num_map.get, '['))])
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        word = to_word(probs_)
        # word = words[np.argmax(probs_)]
        poem = ''
        while word != ']':
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_num_map[word]
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)
        # word = words[np.argmax(probs_)]
        return poem


print(gen_poetry())
