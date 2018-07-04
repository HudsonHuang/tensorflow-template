# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:48:21 2017

@author: TT
"""

import tensorflow as tf
from module.layers import conv2d,max_pool_2x2,weight_variable,bias_variable
from module.loss import reduce_mean_cross_entropy_loss

class deep_mnist(object):
    def __init__(self, hp, x ,y, keep_prob, use_adamW = False):  
        with tf.name_scope("deep_mnist"): 
              
                with tf.name_scope('reshape'):
                    x_image = tf.reshape( x, [-1, 28, 28, 1])
                
                  # First convolutional layer - maps one grayscale image to 32 feature maps.
                with tf.name_scope('conv1'):
                    W_conv1 = weight_variable([5, 5, 1, 32])
                    b_conv1 = bias_variable([32])
                    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
                
                  # Pooling layer - downsamples by 2X.
                with tf.name_scope('pool1'):
                    h_pool1 = max_pool_2x2(h_conv1)
                
                  # Second convolutional layer -- maps 32 feature maps to 64.
                with tf.name_scope('conv2'):
                    W_conv2 = weight_variable([5, 5, 32, 64])
                    b_conv2 = bias_variable([64])
                    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
                
                  # Second pooling layer.
                with tf.name_scope('pool2'):
                    h_pool2 = max_pool_2x2(h_conv2)
                
                  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
                  # is down to 7x7x64 feature maps -- maps this to 1024 features.
                with tf.name_scope('fc1'):
                    W_fc1 = weight_variable([7 * 7 * 64, 1024])
                    b_fc1 = bias_variable([1024])
                
                    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
                    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
                    tf.summary.histogram('h_fc1', h_fc1)   
                
                  # Dropout - controls the complexity of the model, prevents co-adaptation of
                  # features.
                with tf.name_scope('dropout'):
                    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                
                  # Map the 1024 features to 10 classes, one for each digit
                with tf.name_scope('fc2'):
                    W_fc2 = weight_variable([1024, 10])
                    b_fc2 = bias_variable([10])
                
                h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                tf.summary.histogram('h_fc2', h_fc2)   
              
                # Prediction of y(y_hat) and ground_truth label(y)
                self.y_hat=h_fc2
                
                
                with tf.name_scope('cross_entropy_loss'):
                    self.cross_entropy = reduce_mean_cross_entropy_loss(labels= y,
                                                        logits=self.y_hat)
                    tf.summary.scalar('cross_entropy', self.cross_entropy)
                
                with tf.name_scope('adam_optimizer'):
                    if use_adamW:
                        self.train_step = tf.contrib.opt.AdamWOptimizer(weight_decay = 1e-4, learning_rate = hp.learn_rate).minimize(self.cross_entropy)
                    else:
                        self.train_step = tf.train.AdamOptimizer(hp.learn_rate).minimize(self.cross_entropy)
                
                with tf.name_scope('accuracy'):
                    correct_prediction = tf.equal(tf.argmax(self.y_hat, 1), tf.argmax( y, 1))
                    correct_prediction = tf.cast(correct_prediction, tf.float32)
                    self.accuracy = tf.reduce_mean(correct_prediction)
                    tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()