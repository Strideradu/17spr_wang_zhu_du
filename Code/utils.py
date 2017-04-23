#!/usr/bin/env python
#-*- coding:utf-8 -*-

import codecs
import os
import collections
from six.moves import cPickle,reduce,map
import numpy as np

BEGIN_CHAR = '^'
END_CHAR = '$'
UNKNOWN_CHAR = '*'
MAX_VOCAB_SIZE =3000
MAX_TANG_LENGTH = 100
MIN_SONG_LENGTH = 56


class RuleExtractor():
    def __init__(self, selected_cipai=0, encoding="utf-8'"):
        data_dir = '../Data'
        self.encoding = encoding

        #input_file = os.path.join(data_dir,"quansongci_tab.txt")
        input_file = os.path.join(data_dir,"qts_without_tab.txt")

        self.cipai_list = self.get_cipai_list(input_file)
        selected_cipai_index = min(selected_cipai, len(self.cipai_list))
        self.encoding = encoding
        self.cipai_rules = \
            self.get_cipai_rule(input_file,selected_cipai_index)

    def get_cipai_list(self,input_file):
        def extract_cipai(line):
            sentences = line.split()
            return sentences[1]
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            cipai_list = list(map(extract_cipai,f.read().strip().split('\n')))

        cipai_list = list(collections.Counter(cipai_list).items())

        cipai_list = sorted(cipai_list,key=lambda x: x[1],reverse=True)
        cipai_list = [cipai_pair[0] for cipai_pair in cipai_list]
        print("Ci Pai number: "+ str(len(cipai_list)))
        print("Most common cipai: "+ cipai_list[0])

        return cipai_list

    def get_cipai_rule(self, input_file, selected_cipai_index=0):
        #huan xi sha is the default cipai
        selected_cipai = self.cipai_list[selected_cipai_index]

        #### Extracting rules
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            while True:
                sentece = f.readline().strip().split()
                if sentece[1] == selected_cipai:
                        sentece = sentece[3]
                        break
        print("selecting template:")
        print(sentece)

        def extract_rule(sentece):
            punc_list = ["，","。","！","？","、"]
            #punc_list = ['\xef\xbc\x8c',
            #             '\xe3\x80\x82',
            #             '\xef\xbc\x81',
            #             '\xef\xbc\x9f',
            #             '\xe3\x80\x81'
            #             ]
            try:
                punc_list= [item.decode("utf-8") for item in punc_list]
            except:
                pass

            #print(type(punc_list[0]))
            rule_list = [j if j in punc_list else -1 for i,j in enumerate(sentece)]
            return rule_list, punc_list

        rule_list, punc_list = extract_rule(sentece)
        cipai_rule = (selected_cipai, rule_list, punc_list)
        return cipai_rule


class TextLoader():

    def __init__(self, batch_size, cipai=True, max_vocabsize=MAX_VOCAB_SIZE, encoding='utf-8'):
        self.batch_size = batch_size
        self.max_vocabsize = max_vocabsize
        self.encoding = encoding

        data_dir = '../Data'

        input_file = os.path.join(data_dir, "qss_tab.txt")
        input_file = os.path.join(data_dir,"quansongci_tab.txt")
        input_file = os.path.join(data_dir, "qts_without_tab.txt")

        vocab_file = os.path.join(data_dir, "vocab.pkl")
        rhyme_file = os.path.join(data_dir, "rhyme.pkl")
        data_tensor_file = os.path.join(data_dir, "data_tensor.npy")
        rhyme_tensor_file = os.path.join(data_dir, "rhyme_tensor.npy")


        if "qts" not in input_file:
            self.cipai_list = self.get_cipai_list(input_file)
            line_list, self.cipai_rules = self.get_lines_with_specified_cipai(input_file)
        else:
            line_list = None
            self.cipai_rules = None


        ######################################
        # preprocess is the most key function we need to revise. -- By Judy
        self.preprocess(line_list, input_file, vocab_file, rhyme_file, data_tensor_file, rhyme_tensor_file, cipai)
        ######################################


        self.create_batches()
        self.reset_batch_pointer()






    def get_cipai_list(self,input_file):
        def extract_cipai(line):
            sentences = line.split()
            return sentences[1]
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            cipai_list = list(map(extract_cipai,f.read().strip().split('\n')))

        cipai_list = list(collections.Counter(cipai_list).items())

        cipai_list = sorted(cipai_list,key=lambda x: x[1],reverse=True)
        cipai_list = [cipai_pair[0] for cipai_pair in cipai_list]
        print("Ci Pai number: "+ str(len(cipai_list)))
        print("Most common cipai: "+ cipai_list[0])

        return cipai_list



    def get_lines_with_specified_cipai(self, input_file, selected_cipai_index=0):


        #huan xi sha is the default cipai
        selected_cipai = self.cipai_list[selected_cipai_index]

        def keep_line(line,n,selected_cipai):
            sentences = line.split()
            if selected_cipai == sentences[1]:
                return_s = str(n)+' '
            else:
                return_s = ''
            return return_s

        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            poems_list = f.read().strip().split('\n')

        tmp_cipai = [selected_cipai for i in range(len(poems_list))]

        number_list= list(map(keep_line, poems_list, range(len(poems_list)),tmp_cipai))
        number_list = ''.join(number_list)
        number_list = [int(i) for i in number_list.split()]

        number_list = range(len(poems_list))



        #### Extracting rules
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            while True:
                sentece = f.readline().strip().split()
                if sentece[1] == selected_cipai:
                        sentece = sentece[3]
                        break
        print("selecting template:")
        print(sentece)

        def extract_rule(sentece):
            punc_list = ["，","。","！","？","、"]
            #punc_list = ['\xef\xbc\x8c',
            #             '\xe3\x80\x82',
            #             '\xef\xbc\x81',
            #             '\xef\xbc\x9f',
            #             '\xe3\x80\x81'
            #             ]
            try:
                punc_list= [item.decode(self.encoding) for item in punc_list]
            except:
                pass
            #print(type(punc_list[0]))
            rule_list = [j if j in punc_list else -1 for i,j in enumerate(sentece)]
            return rule_list, punc_list

        rule_list, punc_list = extract_rule(sentece)
        cipai_rule = (selected_cipai, rule_list, punc_list)
        return number_list, cipai_rule



    def preprocess(self, line_list, input_file, vocab_file, rhyme_file, data_tensor_file, rhyme_tensor_file, cipai):
        def handle_poem_without_title(line):
            line = line.replace(' ','')
            if len(line) >= MAX_TANG_LENGTH:
                index_end = line.rfind(u'。',0,MAX_TANG_LENGTH)
                index_end = index_end if index_end > 0 else MAX_TANG_LENGTH
                line = line[:index_end+1]
            return BEGIN_CHAR+line+END_CHAR

        def handle_songci_with_title(line):
            sentences = line.split()
            # remove title, only retain cipai and ci content
            sentences = sentences[3:]
            line = ''.join(sentences)
            line = line.replace(' ','')
            return BEGIN_CHAR+line+END_CHAR

        if 'quansongci' in input_file:
            print('Processing Quan Song Ci dataset ..')
            with codecs.open(input_file, "r", encoding=self.encoding) as f:
                poems_list = f.read().strip().split('\n')
            if cipai:
                selected_poems_list = [poems_list[i] for i in line_list]
                lines = list(map(handle_songci_with_title, selected_poems_list))
            else:
                lines = list(map(handle_songci_with_title, poems_list))

        elif 'without' in input_file:  # this is for preprocessing tang shi
            print('Processing Tangshi dataset ..')
            with codecs.open(input_file, "r", encoding=self.encoding) as f:
                lines = list(map(handle_poem_without_title,f.read().strip().split('\n')))

        else:
            pass

        print("Number of Selected Song Ci:" + str(len(lines)))


        # counter: similar to a dictionary {word:occurence count} --By Judy
        counter = collections.Counter(reduce(lambda data,line: line+data,lines,''))
        # counter_pairs: sorted list [(word:occurence count)] in decreasing
        # order -- By Judy
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        # chars: a tuple of unique Chinese characters, in order of decreasing
        # occurence count. -- By Judy
        chars, _ = zip(*count_pairs)

        # vocab_size: word_count + 1, and 1 is for '*'. --By Judy
        self.vocab_size = min(len(chars),self.max_vocabsize - 1) + 1

        # chars: a tuple, updated, and the last elment is '*'. --By Judy
        self.chars = chars[:self.vocab_size-1] + (UNKNOWN_CHAR,)

        # vocab: a dict {word:ID}. iD is assigned by order of decreasing occurence
        # count. -- By Judy
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        ## get rhyme
        from pypinyin import lazy_pinyin
        self.rhymes = dict()
        rhyme_set = set()
        for word, ID in self.vocab.items():
            #print(type(word),word)
            try:
                py = lazy_pinyin(word)
                py = py[0].encode("utf-8")
                vow_pos=[1 if i in 'aeiou' else -1 for i in py]
                try:
                    vow_start_index = vow_pos.index(1)
                except:
                    vow_start_index = len(vow_pos)
                rhyme = py[vow_start_index:]
            except:
                rhyme = ''
            self.rhymes[word] = rhyme
            rhyme_set.add(rhyme)
        rhyme_set = list(rhyme_set)

        for word, rhyme in self.rhymes.items():
            self.rhymes[word] = rhyme_set.index(rhyme)


        unknown_char_int = self.vocab.get(UNKNOWN_CHAR)
        unknown_rhyme_int = self.rhymes.get(UNKNOWN_CHAR)

        # get int: a temporary function which returns the ID of an input word.
        # If the word does not exist, return the ID of '*'. --By Judy
        get_int = lambda char: self.vocab.get(char,unknown_char_int)
        get_rhyme = lambda char: self.rhymes.get(char, unknown_rhyme_int)

        lines = sorted(lines,key=lambda line: len(line))

        # data_tensor: a list of sentences. in each sentence, the character is
        # transformed to its associated ID. --By Judy
        self.data_tensor = [ list(map(get_int,line)) for line in lines ]

        self.rhyme_tensor = [ list(map(get_rhyme,line)) for line in lines ]

        self.poem_length = max(map(len,self.data_tensor))

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        with open(rhyme_file, 'wb') as f:
            cPickle.dump(self.rhymes, f)
        with open(data_tensor_file,'wb') as f:
            cPickle.dump(self.data_tensor,f)
        with open(rhyme_tensor_file, 'wb') as f:
            cPickle.dump(self.rhyme_tensor,f)


    def load_preprocessed(self, vocab_file, data_tensor_file, rhyme_file, rhyme_tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        with open(data_tensor_file,'rb') as f:
            self.data_tensor = cPickle.load(f)
        with open(rhyme_tensor_file, 'rb') as f:
            self.rhyme_tensor = cPickle.load(f)
        with open(rhyme_file, 'rb') as f:
            self.rhymes = cPickle.load(f)

        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

    def create_batches(self):
        self.num_batches = int(len(self.data_tensor) / self.batch_size)
        self.data_tensor = self.data_tensor[:self.num_batches * self.batch_size]
        self.rhyme_tensor = self.rhyme_tensor[:self.num_batches * self.batch_size]

        unknown_char_int = self.vocab.get(UNKNOWN_CHAR)
        unknown_rhyme_int = self.rhymes.get(UNKNOWN_CHAR)

        self.xdata_batches = []
        self.xrhyme_batches = []

        self.ydata_batches = []
        self.yrhyme_batches = []

        for i in range(self.num_batches):
            from_index = i * self.batch_size
            to_index = from_index + self.batch_size
            # batches: batch_size poems
            data_batches = self.data_tensor[from_index:to_index]
            rhyme_batches = self.rhyme_tensor[from_index:to_index]

            # seq_length: number of characters in the longest poem in batches
            #seq_length = max(map(len,data_batches))
            seq_length = self.poem_length

            # xdata: a matrix of size batch_size X seq_length, inital valuse =
            # unknown_char_int
            xdata = np.full((self.batch_size,seq_length),unknown_char_int,np.int32)
            xrhyme = np.full((self.batch_size,seq_length),unknown_rhyme_int,np.int32)

            for row in range(self.batch_size):
                xdata[row,:len(data_batches[row])] = data_batches[row]
                xrhyme[row,:len(rhyme_batches[row])] = rhyme_batches[row]

            ydata = np.copy(xdata)
            ydata[:,:-1] = xdata[:,1:]

            yrhyme = np.copy(xrhyme)
            yrhyme[:,:-1] = xrhyme[:,1:]

            self.xdata_batches.append(xdata)
            self.xrhyme_batches.append(xrhyme)

            self.ydata_batches.append(ydata)
            self.yrhyme_batches.append(yrhyme)

    def next_batch(self, rhyme=True):
        xdata, ydata = self.xdata_batches[self.pointer], self.ydata_batches[self.pointer]
        xrhyme, yrhyme = self.xrhyme_batches[self.pointer], self.yrhyme_batches[self.pointer]
        self.pointer += 1
        if rhyme == True:
            return xdata, ydata, xrhyme, yrhyme
        else:
            return xdata, ydata

    def reset_batch_pointer(self):
        self.pointer = 0
