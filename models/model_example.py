# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:47:44 2017

@author: TT
"""
import tensorflow as tf
from module import layers

class model_example(object):
    def __init__(self, hp):  
        with tf.name_scope("model_example"): 
              self.x = tf.placeholder(tf.float32, [None, hp.input_dim])
              W = tf.Variable(tf.zeros([hp.input_dim, hp.input_dim]))
              b = tf.Variable(tf.zeros([hp.input_dim]))
              self.inputs = tf.matmul(self.x, W) + b
              
              #use For loop to allocate more layers
              #Or any conditional branch to make computational graph dynamically
              for i in range(len(hp.hidden_units)):
                  self.inputs=layers.softmax_layers(self.inputs,hp.hidden_units[i])
              
            
              # Define loss and optimizer
              self.y_ = tf.placeholder(tf.float32, [None, hp.output_dim])
            
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
                          labels=self.y_, logits=self.inputs))
              tf.summary.scalar('cross_entropy', self.cross_entropy)
              self.train_step = tf.train.GradientDescentOptimizer(hp.lr).minimize(
                      self.cross_entropy)
              
    def model_eval(self):
        with tf.name_scope("model_example_eval"): 
            correct_prediction = tf.equal(tf.argmax(self.inputs, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy