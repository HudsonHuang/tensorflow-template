# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:47:44 2017

@author: TT
"""
import tensorflow as tf
from module import layers

class model_example(object):
    def __init__(self, hp,x,y):  
        with tf.name_scope("model_example"): 
#              self.x = tf.placeholder(tf.float32, [None, hp.input_dim])
              self.inputs = x
              
              #use For loop to allocate more layers
              #Or any conditional branch to make computational graph dynamically
              for i in range(len(hp.hidden_units)):
                  self.inputs=layers.softmax_layers(self.inputs,hp.hidden_units[i])
              
              # Prediction of y(y_hat) and ground_truth label(y)
              self.y_hat=self.inputs
#              self.y = tf.placeholder(tf.float32, [None, hp.output_dim])
            
              # Define loss and optimizer
              self.cross_entropy = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits(
                          labels=y, logits=self.y_hat))
              tf.summary.scalar('cross_entropy', self.cross_entropy)
              self.train_step = tf.train.GradientDescentOptimizer(hp.lr).minimize(
                      self.cross_entropy)
              
              correct_prediction = tf.equal(tf.argmax(self.y_hat, 1), tf.argmax(y, 1))
              self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
              tf.summary.scalar('accuracy', self.accuracy)
              
        self.merged = tf.summary.merge_all()