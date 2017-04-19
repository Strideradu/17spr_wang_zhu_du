#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import argparse
import os
from six.moves import cPickle

from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('--prime', type=str, default='',
                       help=u'输入指定文字生成藏头诗')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep')

    args = parser.parse_args()
    sample(args)

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
        args.cipai_rules = saved_args.cipai_rules
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
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
