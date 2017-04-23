#-*- coding: utf-8 -*-

with open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master-backup/chinese-poem-generator-master/data/qsc.txt', 'r') as data:
    sentence_num = 0
    while 1:
        line = data.readline()
        line = line.strip().decode('utf-8')
        if not line:
            continue
        if line == 'END':
            break
        if (u"，" not in line) and (u"。" not in line):
            sentence_num += 1
print 'sentence_num', sentence_num


inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/bigram_count_dict', 'r')
bigram_count_dict = {}
bigram_num = 0
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split(', "')

for i in array:
    tem_array = i.strip().replace('"', '').split(':')
    if '\u' in tem_array[0]:
        bigram = tem_array[0].strip().decode('unicode-escape').encode('utf-8')
        bigram_count_dict[bigram] = int(tem_array[1].strip())
        bigram_num += 1
sorted_bigram_count_dict = sorted(
    bigram_count_dict.iteritems(), key=lambda d: d[1], reverse=True)
# for i in sorted_bigram_count_dict:
#     print i,sorted_bigram_count_dict[i]
print bigram_num

inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/bigram_word_to_end_dict', 'r')
bigram_word_to_end_dict = {}
bigram_num = 0
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split('], ')
for i in array:
    tem_array = i.strip().replace('"', '').replace(
        ']', '').replace('[', '').split(':')
    if '\u' in tem_array[0]:
        bigram = tem_array[0].strip().decode('unicode-escape').encode('utf-8')
        bigram_word_to_end_dict[bigram] = tem_array[1].strip().decode(
            'unicode-escape').encode('utf-8').split(', ')

# for i in bigram_word_to_end_dict:
#     print i, bigram_word_to_end_dict[i]


inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/bigram_word_to_start_dict', 'r')
bigram_word_to_start_dict = {}
bigram_num = 0
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split('], ')
for i in array:
    tem_array = i.strip().replace('"', '').replace(
        ']', '').replace('[', '').split(':')
    if '\u' in tem_array[0]:
        bigram = tem_array[0].strip().decode('unicode-escape').encode('utf-8')
        bigram_word_to_start_dict[bigram] = tem_array[1].strip().decode(
            'unicode-escape').encode('utf-8').split(', ')

# for i in bigram_word_to_start_dict:
#     print i, bigram_word_to_start_dict[i]
#

inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/pingze_rhythm_dict', 'r')
pingze_rhythm_dict = {}
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split('], ')

for i in array:
    tem_array = i.strip().replace('"', '').replace(
        ']', '').replace('[', '').split(':')
    if '\u' not in tem_array[0]:
        pingze = int(tem_array[0].strip())
        pingze_rhythm_dict[pingze] = tem_array[1].strip().decode(
            'unicode-escape').encode('utf-8').split(', ')

# for i in pingze_rhythm_dict:
#     print i, pingze_rhythm_dict[i]


inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/pingze_words_dict', 'r')
pingze_words_dict = {}
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split('], ')
    # print line[0:1000]

for i in array:
    tem_array = i.strip().replace('"', '').replace(
        ']', '').replace('[', '').split(':')
    if '\u' not in tem_array[0]:
        pingze = int(tem_array[0].strip())
        pingze_words_dict[pingze] = tem_array[1].strip().decode(
            'unicode-escape').encode('utf-8').split(', ')

# for i in pingze_words_dict:
#     print i, pingze_words_dict[i]


inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/reverse_pingze_word_dict', 'r')
reverse_pingze_word_dict = {}
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split(', ')
    # print line[0:1000]

for i in array:
    tem_array = i.strip().replace('"', '').split(':')
    if '\u' in tem_array[0]:
        word = tem_array[0].strip().decode('unicode-escape').encode('utf-8')
        reverse_pingze_word_dict[word] = int(tem_array[1].strip())

# for i in reverse_pingze_word_dict:
#     print i, reverse_pingze_word_dict[i]
#

inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/reverse_rhythm_word_dict', 'r')
reverse_rhythm_word_dict = {}
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split(', ')
    # print line[0:1000]

for i in array:
    tem_array = i.strip().replace('"', '').split(':')
    if '\u' in tem_array[0]:
        word = tem_array[0].strip().decode('unicode-escape').encode('utf-8')
        reverse_rhythm_word_dict[word] = tem_array[
            1].strip().decode('unicode-escape').encode('utf-8')

# for i in reverse_rhythm_word_dict:
#     print i, reverse_rhythm_word_dict[i]


inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/rhythm_count_dict', 'r')
rhythm_count_dict = {}
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split(', ')
    # print line[0:1000]

for i in array:
    tem_array = i.strip().replace('"', '').split(':')
    if '\u' in tem_array[0]:
        word = tem_array[0].strip().decode('unicode-escape').encode('utf-8')
        rhythm_count_dict[word] = int(tem_array[1].strip())
sorted_rhythm_count_dict = sorted(
    rhythm_count_dict.iteritems(), key=lambda d: d[1], reverse=True)
# for i in sorted_rhythm_count_dict:
#     print i[0], i[1]


inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/rhythm_word_dict', 'r')
rhythm_word_dict = {}
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split('], ')
    # print line[0:1000]

for i in array:
    tem_array = i.strip().replace('"', '').replace(
        ']', '').replace('[', '').split(':')
    if '\u' in tem_array[0]:
        word = tem_array[0].strip().decode('unicode-escape').encode('utf-8')
        rhythm_word_dict[word] = tem_array[1].strip().decode(
            'unicode-escape').encode('utf-8').split(', ')
# for i in rhythm_word_dict:
#     print i, rhythm_word_dict[i]


inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/title_delimiter_dict', 'r')
title_delimiter_dict = {}
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split('], ')
    # print line[0:1000]

for i in array:
    tem_array = i.strip().replace('"', '').replace(
        ']', '').replace('[', '').split(':')
    if '\u' in tem_array[0]:
        word = tem_array[0].strip().decode('unicode-escape').encode('utf-8')
        title_delimiter_dict[word] = tem_array[1].strip().decode(
            'unicode-escape').encode('utf-8').split(', ')
# for i in title_delimiter_dict:
#     print i, title_delimiter_dict[i]


inf = open('D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/title_pingze_dict', 'r')
title_pingze_dict = {}
for line in inf.readlines():
    line = line.strip()[1:-1]
    array = line.split('], ')
    # print line[0:1000]

for i in array:
    tem_array = i.strip().replace('"', '').replace(
        ']', '').replace('[', '').split(':')
    if '\u' in tem_array[0]:
        word = tem_array[0].strip().decode('unicode-escape').encode('utf-8')
        title_pingze_dict[word] = tem_array[1].strip().decode(
            'unicode-escape').encode('utf-8').split(', ')
for i in title_pingze_dict:
    print i, title_pingze_dict[i]
