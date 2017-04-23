#-*- coding: utf-8 -*-
# vim: set bg=dark noet ts=4 sw=4 fdm=indent :

# # pingze_rhythm_dict
# infile = 'D:\Course 2017 Spring\CSE847\Project_Related\chinese-poem-generator-master\data\pingze_rhythm_dict'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split('"')
#
# pingze_rhythm_dict = {}
# for i in array:
#     if '{' not in i and '}' not in i:
#         if len(i) == 1:
#             index = int(i.strip())
#             pingze_rhythm_dict[index] = []
#         elif '\u' in i:
#             item = i.strip().decode('unicode-escape').encode('utf-8')
#             pingze_rhythm_dict[index].append(item)
# for i in pingze_rhythm_dict:
#     print i
#     for j in pingze_rhythm_dict[i]:
#         print j,
#     print '\n'


# # reverse_rhythm_word_dict
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/reverse_rhythm_word_dict'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split(',')
#
# pingze_rhythm_dict = {}
# index = 0
# for i in array:
#     tem_array = i.strip().split('"')
#     key = tem_array[1].strip().decode('unicode-escape').encode('utf-8')
#     value = tem_array[3].strip().decode('unicode-escape').encode('utf-8')
#     pingze_rhythm_dict[index] = [key, value]
#     index += 1
# for i in pingze_rhythm_dict:
#     print i
#     for j in pingze_rhythm_dict[i]:
#         print j,
#     print '\n'


# # reverse_pingze_word_dict
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/reverse_pingze_word_dict'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split(',')
#
# pingze_rhythm_dict = {}
# index = 0
# for i in array:
#     tem_array = i.strip().split('"')
#     key = tem_array[1].strip().decode('unicode-escape').encode('utf-8')
#     value = tem_array[3].strip().decode('unicode-escape').encode('utf-8')
#     pingze_rhythm_dict[index] = [key, value]
#     index += 1
# for i in pingze_rhythm_dict:
#     print i
#     for j in pingze_rhythm_dict[i]:
#         print j,
#     print '\n'


# # rhythm_word_dict
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/rhythm_word_dict'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split('],')
#
# pingze_rhythm_dict = {}
# index = 0
# for i in array:
#     tem_array = i.strip().split('"')
#     key = tem_array[1].strip().decode('unicode-escape').encode('utf-8')
#     pingze_rhythm_dict[key] = []
#     for j in range(len(tem_array)):
#         if '\u' in tem_array[j] and j != 1:
#             value = tem_array[j].strip().decode(
#                 'unicode-escape').encode('utf-8')
#             pingze_rhythm_dict[key].append(value)
#
# for i in pingze_rhythm_dict:
#     print i
#     for j in pingze_rhythm_dict[i]:
#         print j,
#     print '\n'


# # pingze_words_dict
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/pingze_words_dict'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split('],')
#
# pingze_rhythm_dict = {}
# index = 0
# for i in array:
#     tem_array = i.strip().split('"')
#     key = tem_array[1].strip().decode('unicode-escape').encode('utf-8')
#     pingze_rhythm_dict[key] = []
#     for j in range(len(tem_array)):
#         if '\u' in tem_array[j] and j != 1:
#             value = tem_array[j].strip().decode(
#                 'unicode-escape').encode('utf-8')
#             pingze_rhythm_dict[key].append(value)
#
# for i in pingze_rhythm_dict:
#     print i
#     for j in pingze_rhythm_dict[i]:
#         print j,
#     print '\n'

#
# # sentences
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/sentences'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split(',')
#
# sentences = []
# index = 0
# for i in array:
#     tem_array = i.strip().split('"')
#     for j in tem_array:
#         if '\u' in j:
#             sentences.append(j.strip().decode(
#                 'unicode-escape').encode('utf-8'))
# for i in sentences:
#     print i


# # word_count_dict
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/word_count_dict'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split(',')
#
# word_count_dict = {}
# for i in array:
#     tem_array = i.strip().split(':')
#     key = tem_array[0].strip()[1:-1].decode('unicode-escape').encode('utf-8')
#     word_count_dict[key] = int(tem_array[1].strip().split('}')[0])
#
#
# for i in word_count_dict:
#     print i,word_count_dict[i]


# # bigram_count_dict
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/bigram_count_dict'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split(',')
# print len(array)
#
# bigram_count_dict = {}
# for i in array:
#     tem_array = i.strip().split(':')
#     key = tem_array[0].strip()[1:-1].decode('unicode-escape').encode('utf-8')
#     # print tem_array
#     if len(tem_array)>1:
#         if '}' in tem_array[1]:
#             continue
#         if tem_array[1] !='':
#             bigram_count_dict[key] = int(tem_array[1].strip())
#
# dict= sorted(bigram_count_dict.iteritems(), key=lambda d:d[1], reverse = True)
# for i in dict:
#     print i[0],i[1]


# # split_sentences
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/split_sentences'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     line=i.strip().replace('"','').replace(']','').replace('[','')
#     array = line.strip().split(',')
#
#
# split_sentences = []
# for i in array:
#     if '\u' in i:
#         split_sentences.append(i.strip().decode('unicode-escape').encode('utf-8'))
# for i in split_sentences:
#     print i


# # rhythm_count_dict
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/rhythm_count_dict'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split(',')
# print len(array)
#
# rhythm_count_dict = {}
# for i in array:
#     tem_array = i.strip().split(':')
#     key = tem_array[0].strip()[1:-1].decode('unicode-escape').encode('utf-8')
#     # print tem_array
#     if len(tem_array) > 1:
#         if '}' in tem_array[1]:
#             continue
#         if tem_array[1] != '':
#             rhythm_count_dict[key] = int(tem_array[1].strip())
#
# dict = sorted(rhythm_count_dict.iteritems(), key=lambda d: d[1], reverse=True)
# for i in dict:
#     print i[0], i[1]


# # title_pingze_dict
# infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/title_pingze_dict'
# inf = open(infile, 'r')
# for i in inf.readlines():
#     array = i.strip().split('"')
# print len(array)
#
# title_pingze_dict = {}
# for i in array:
#     if '\u' in i:
#         cipai = i.strip().decode(
#             'unicode-escape').encode('utf-8')
#         title_pingze_dict[cipai] = []
#     elif ('0' in i) or ('1' in i) or ('2' in i):
#         title_pingze_dict[cipai].append(i.strip())
# for i in title_pingze_dict:
#     print i
#     print title_pingze_dict[i]


# title_delimiter_dict
infile = 'D:/Course 2017 Spring/CSE847/Project_Related/chinese-poem-generator-master/data/title_delimiter_dict'
inf = open(infile, 'r')
for i in inf.readlines():
    array = i.strip().split('"')
print len(array)

title_delimiter_dict = {}
deli_list = [',', '.', '|', '`']
for i in array:
    tem = i.strip()
    if '\u' in tem:
        cipai = tem.decode('unicode-escape').encode('utf-8')
        title_delimiter_dict[cipai] = []
    elif tem in deli_list:
        title_delimiter_dict[cipai].append(tem)

for i in title_delimiter_dict:
    print i, title_delimiter_dict[i]
