from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Poem_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []
        self.word2idx = {}

    def create_batches(self, data_file, seq_length):
        self.token_stream = []
        self.token = [] # token for generate word to id dict
        token_text = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                #line = line.split()
                self.token.extend(list(line))
                # parse_line = [int(x) for x in line]
                # print(line)
                # print(len(line))
                if len(line) % seq_length == 0:
                    while True:
                        token_text.append(list(line[0:seq_length]))
                        if len(line) == seq_length:
                            break
                        else:
                            line = line[seq_length:]

        print("Found tokens: ", len(token_text))
        self.words = ['_START'] + list(set(self.token))
        self.word2idx = dict((word, i) for i, word in enumerate(self.words))
        for token in token_text:
            self.token_stream.append([self.word2idx[tok] for tok in token])
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0
        return (len(self.words))

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

    def get_words(self):
        return self.words

    def get_word2idx(self):
        return self.word2idx

class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file, seq_length):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == seq_length:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file, seq_length, word2idx):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                #line = line.split()
                #parse_line = [int(x) for x in line]
                if len(line) % seq_length == 0:
                    while True:
                        positive_examples.append([word2idx[tok] for tok in list(line[0:seq_length])])
                        if len(line) == seq_length:
                            break
                        else:
                            line = line[seq_length:]

        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == seq_length:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0