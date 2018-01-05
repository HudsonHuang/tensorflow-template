# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:10:40 2018

@author: TT
"""

import tensorflow as tf

def reduce_mean_cross_entropy_loss(labels,logits):
    with tf.variable_scope("reduce_mean_cross_entropy_loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels,
                                                                logits = logits)
        cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy