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
MAX_VOCAB_SIZE =600
MAX_TANG_LENGTH = 100
MIN_SONG_LENGTH = 56

class TextLoader():

    def __init__(self, batch_size, max_vocabsize=MAX_VOCAB_SIZE, encoding='utf-8'):
        self.batch_size = batch_size
        self.max_vocabsize = max_vocabsize
        self.encoding = encoding

        data_dir = '../Data'

        input_file = os.path.join(data_dir, "qts_without_tab.txt")
        input_file = os.path.join(data_dir, "qss_tab.txt")
        input_file = os.path.join(data_dir,"quansongci_tab.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        self.cipai_list = self.get_cipai_list(input_file)

        line_list, self.cipai_rules = self.get_lines_with_specified_cipai(input_file)


        ######################################
        # preprocess is the most key function we need to revise. -- By Judy
        self.preprocess(line_list, input_file, vocab_file, tensor_file)
        ######################################

        #if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
        #    print("reading text file")
        #    self.preprocess(line_list, input_file, vocab_file, tensor_file)
        #else:
        #    print("loading preprocessed files")
        #    self.load_preprocessed(vocab_file, tensor_file)

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
            #punc_list = ["，","。","！","？","、"]
            punc_list = ['\xef\xbc\x8c',
                         '\xe3\x80\x82',
                         '\xef\xbc\x81',
                         '\xef\xbc\x9f',
                         '\xe3\x80\x81'
                         ]
            punc_list= [item.decode("utf-8") for item in punc_list]
            #print(type(punc_list[0]))
            rule_list = [j if j in punc_list else -1 for i,j in enumerate(sentece)]
            return rule_list, punc_list

        rule_list, punc_list = extract_rule(sentece)
        cipai_rule = (selected_cipai, rule_list, punc_list)
        return number_list, cipai_rule



    def preprocess(self, line_list, input_file, vocab_file, tensor_file):
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
            selected_poems_list = [poems_list[i] for i in line_list]
            lines = list(map(handle_songci_with_title, selected_poems_list))

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

        unknown_char_int = self.vocab.get(UNKNOWN_CHAR)
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)

        # get int: a temporary function which returns the ID of an input word.
        # If the word does not exist, return the ID of '*'. --By Judy
        get_int = lambda char: self.vocab.get(char,unknown_char_int)
        lines = sorted(lines,key=lambda line: len(line))

        # tensor: a list of sentences. in each sentence, the character is
        # transformed to its associated ID. --By Judy
        self.tensor = [ list(map(get_int,line)) for line in lines ]
        with open(tensor_file,'wb') as f:
            cPickle.dump(self.tensor,f)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        with open(tensor_file,'rb') as f:
            self.tensor = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

    def create_batches(self):
        self.num_batches = int(len(self.tensor) / self.batch_size)
        self.tensor = self.tensor[:self.num_batches * self.batch_size]
        unknown_char_int = self.vocab.get(UNKNOWN_CHAR)
        self.x_batches = []
        self.y_batches = []

        for i in range(self.num_batches):
            from_index = i * self.batch_size
            to_index = from_index + self.batch_size
            # batches: batch_size poems
            batches = self.tensor[from_index:to_index]

            # seq_length: number of characters in the longest poem in batches
            seq_length = max(map(len,batches))

            # xdata: a matrix of size batch_size X seq_length, inital valuse =
            # unknown_char_int
            xdata = np.full((self.batch_size,seq_length),unknown_char_int,np.int32)
            for row in range(self.batch_size):
                xdata[row,:len(batches[row])] = batches[row]

            ydata = np.copy(xdata)
            ydata[:,:-1] = xdata[:,1:]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
