#-*- coding: utf-8 -*-
# vim: set bg=dark noet ts=4 sw=4 fdm=indent :
import json
import jieba
import jieba.posseg as pseg
from gensim import models

word_model = models.Word2Vec.load(
    'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/word_model')
similar_words = word_model.most_similar(positive=[u'落花'], topn=20)

# # print similar owrds
# for i in similar_words:
#     for j in i:
#         if type(j) != float:
#             j = j.encode('unicode-escape')
#             print j.decode('unicode-escape').encode('utf-8'),
#         else:
#             print j

filtered_similar_words = []
for (word, similarity) in similar_words:
    word_elems = pseg.cut(word)
    word_flag_valid = False
    for word_elem, flag in word_elems:
        if flag in ['n', 'ns', 'nr', 't']:
            word_flag_valid = True
            break

    if len(word) < 2 and (not word_flag_valid):
        continue

    filtered_similar_words.append((word, similarity))
    print word.encode('unicode-escape').decode('unicode-escape').encode('utf-8')
    print similarity
