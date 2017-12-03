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

from models.model_example import model_example

FLAGS = None

def main(_):
  # Prepare data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  # Create the model
  model=model_example(n_layers=2)

  #Make tf.summary for tensorboard
  merged = tf.summary.merge_all()
  #选定可视化存储目录  
  train_writer = tf.summary.FileWriter("./logdir",model.train_step.graph)  

  #Start tf session
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  
  # Train
  for steps in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary = sess.run(model.train_step, feed_dict={model.x: batch_xs, model.y_: batch_ys})
    train_writer.add_summary(summary, steps)
    if steps % 20 ==0:
    #Evaluate
          correct_prediction = tf.equal(tf.argmax(model.y, 1), tf.argmax(model.y_, 1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          print(sess.run(accuracy, feed_dict={model.x: mnist.test.images,
                                              model.y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default="./datasets/MNIST_data/")
  parser.add_argument('--log_dir', type=str, default="./generated/logdir/")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)