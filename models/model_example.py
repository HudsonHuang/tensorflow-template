# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:47:44 2017

@author: TT
"""
import tensorflow as tf

class model_example(object):
    def __init__(self, len_x=1, n_layers=1):  
        with tf.name_scope("model_example"): 
              self.x = tf.placeholder(tf.float32, [None, 784])
              W = tf.Variable(tf.zeros([784, 10]))
              b = tf.Variable(tf.zeros([10]))
              self.y = tf.matmul(self.x, W) + b
              
              #use For loop to allocate more layers
              #Or any conditional branch to make computational graph dynamically
              for i in range(n_layers):
                  self.y=model_example.more_layers(self.y)
              
            
              # Define loss and optimizer
              self.y_ = tf.placeholder(tf.float32, [None, 10])
            
              # The raw formulation of cross-entropy,
              #
              #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
              #                                 reduction_indices=[1]))
              #
              # can be numerically unstable.
              #
              # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
              # outputs of 'y', and then average across the batch.
              self.cross_entropy = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits(
                          labels=self.y_, logits=self.y))
              self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
                      self.cross_entropy)

    def more_layers(y):
        W = tf.Variable(tf.zeros([10]))
        b = tf.Variable(tf.zeros([10]))
        y = y*W + b
        y = tf.nn.relu(y)
        return y