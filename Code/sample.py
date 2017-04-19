#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import argparse
import os
from six.moves import cPickle

from model import Model
from utils import RuleExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('--prime', type=str, default='',
                       help=u'输入指定文字生成藏头诗')
    parser.add_argument('--cipai', type= int, default=0,
                        help='Index of selected cipai, 0 is the most frequenct cipai')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep')

    args = parser.parse_args()
    sample(args)

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    rule_extractor = RuleExtractor(args.cipai)
    args.cipai_rules = rule_extractor.cipai_rules 
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess,
                               chars,
                               vocab,
                               args.prime,
                               args.sample,
                               args.cipai_rules),"\n")

if __name__ == '__main__':
    main()
