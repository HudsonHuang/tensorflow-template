"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

class ole(object):
    def __init__(self, len_x=1, n_layers=1):  
        with tf.name_scope("ole"): 
              self.x = tf.placeholder(tf.float32, [None, 784])
              W = tf.Variable(tf.zeros([784, 10]))
              b = tf.Variable(tf.zeros([10]))
              self.y = tf.matmul(self.x, W) + b
              
              for i in range(n_layers):
                  self.y=ole.more_layers(self.y)
              
            
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
                  tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
              self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

    def more_layers(y):
        W = tf.Variable(tf.zeros([10]))
        b = tf.Variable(tf.zeros([10]))
        y = y*W + b
        y = tf.nn.relu(y)
        return y

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model

  ole_model=ole(n_layers=2)

  #合并到Summary中  
  merged = tf.summary.merge_all()
  #选定可视化存储目录  
  train_writer = tf.summary.FileWriter("./logdir",ole_model.train_step.graph)  

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  
  # Train
  for steps in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary = sess.run(ole_model.train_step, feed_dict={ole_model.x: batch_xs, ole_model.y_: batch_ys})
    train_writer.add_summary(summary, steps)
    if steps % 20 ==0:
          correct_prediction = tf.equal(tf.argmax(ole_model.y, 1), tf.argmax(ole_model.y_, 1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          print(sess.run(accuracy, feed_dict={ole_model.x: mnist.test.images,
                                              ole_model.y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default="MNIST_data/")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)