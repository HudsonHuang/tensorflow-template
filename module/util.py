# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:58:55 2018

@author: TT
"""
import tensorflow as tf

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)