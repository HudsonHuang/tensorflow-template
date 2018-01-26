# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:48:21 2017

@author: TT
"""

import tensorflow as tf
from module.layers import conv2d,max_pool_2x2,weight_variable,bias_variable
from module.loss import reduce_mean_cross_entropy_loss

class deep_mnist_with_Res(object):
    def __init__(self, hp, x ,y, keep_prob):  
        with tf.name_scope("deep_mnist_with_Res_9conv"): 
              
                with tf.name_scope('reshape'):
                    x_image = tf.reshape( x, [-1, 28, 28, 1])
                
                  # First convolutional layer - maps one grayscale image to 32 feature maps.
                with tf.name_scope('conv0'):
                    W_conv0 = weight_variable([5, 5, 1, 8])
                    b_conv0 = bias_variable([8])
                    h_conv0 = tf.nn.relu(conv2d(x_image, W_conv0) + b_conv0)
                    
                with tf.name_scope('conv0_1'):
                    W_conv0_1 = weight_variable([5, 5, 8, 8])
                    b_conv0_1 = bias_variable([8])
                    h_conv0_1 = tf.nn.relu(conv2d(h_conv0, W_conv0_1) + b_conv0_1)
                    
                with tf.name_scope('conv1'):
                    W_conv1 = weight_variable([5, 5, 8, 16])
                    b_conv1 = bias_variable([16])
                    h_conv1 = tf.nn.relu(conv2d(h_conv0_1, W_conv1) + b_conv1)
                    
                with tf.name_scope('conv1_1'):
                    W_conv1_1 = weight_variable([5, 5, 16, 16])
                    b_conv1_1 = bias_variable([16])
                    h_conv1_1 = tf.nn.relu(conv2d(h_conv1, W_conv1_1) + b_conv1_1)
                
                  # Pooling layer - downsamples by 2X.
                with tf.name_scope('pool1'):
                    h_pool1 = max_pool_2x2(h_conv1_1)
                
                  # Second convolutional layer -- maps 32 feature maps to 64.
                with tf.name_scope('conv2'):
                    W_conv2 = weight_variable([5, 5, 16, 32])
                    b_conv2 = bias_variable([32])
                    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
                    
                  # Second convolutional layer -- maps 32 feature maps to 64.
                with tf.name_scope('conv2_1'):
                    W_conv2_1 = weight_variable([5, 5, 32, 32])
                    b_conv2_1 = bias_variable([32])
                    h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1)

                  # Second convolutional layer -- maps 32 feature maps to 64.
                with tf.name_scope('conv3'):
                    W_conv3 = weight_variable([5, 5, 32, 64])
                    b_conv3 = bias_variable([64])
                    h_conv3 = tf.nn.relu(conv2d(h_conv2_1, W_conv3) + b_conv3)
                
                  # Second convolutional layer -- maps 32 feature maps to 64.
                with tf.name_scope('conv3_1'):
                    W_conv3_1 = weight_variable([5, 5, 64, 64])
                    b_conv3_1 = bias_variable([64])
                    h_conv3_1 = tf.nn.relu(conv2d(h_conv3, W_conv3_1) + b_conv3_1)
                
                  # Second convolutional layer -- maps 32 feature maps to 64.
                with tf.name_scope('conv4'):
                    W_conv4 = weight_variable([5, 5, 64, 128])
                    b_conv4 = bias_variable([128])
                    h_conv4 = tf.nn.relu(conv2d(h_conv3_1, W_conv4) + b_conv4)
                
                  # Second convolutional layer -- maps 32 feature maps to 64.
                with tf.name_scope('conv5'):
                    W_conv5 = weight_variable([5, 5, 128, 64])
                    b_conv5 = bias_variable([64])
                    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
                    
                  # Second convolutional layer -- maps 32 feature maps to 64.
                with tf.name_scope('conv5_1'):
                    W_conv5_1 = weight_variable([5, 5, 64, 64])
                    b_conv5_1 = bias_variable([64])
                    h_conv5_1 = tf.nn.relu(conv2d(h_conv5, W_conv5_1) + b_conv5_1)
                
                with tf.name_scope('res_connect'):
                    res_connect = h_conv3_1 + h_conv5_1
                
                  # Second pooling layer.
                with tf.name_scope('pool2'):
                    h_pool2 = max_pool_2x2(res_connect)
                
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
                
                
                with tf.name_scope('loss'):
                    self.cross_entropy = reduce_mean_cross_entropy_loss(labels= y,
                                                        logits=self.y_hat)
                    tf.summary.scalar('cross_entropy', self.cross_entropy)
                
                with tf.name_scope('adam_optimizer'):
                    self.train_step = tf.train.AdamOptimizer(hp.learn_rate).minimize(self.cross_entropy)
                
                with tf.name_scope('accuracy'):
                    correct_prediction = tf.equal(tf.argmax(self.y_hat, 1), tf.argmax( y, 1))
                    correct_prediction = tf.cast(correct_prediction, tf.float32)
                    self.accuracy = tf.reduce_mean(correct_prediction)
                    tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()