"""
Function of main.py:

config loader
json loader

Call feature extraction
Call model training and validation
Model Save and Load
Call model validation

载入训练参数
载入指定模型超参数的json

调用特征提取
调用模型训练和验证
模型保存与载入
调用模型验证

This example was modified from official site of Tensorflow.
"""
"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""

import argparse
import sys
import datetime
from tqdm import tqdm

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from models.model_example import model_example
from models.deep_mnist import deep_mnist

import params 

FLAGS = None

def prepare_params():
    if FLAGS.experiment_name == "default":
        now=datetime.datetime.now()
        FLAGS.experiment_name=now.strftime('%Y%m%d%H%M%S')
    FLAGS.log_dir = FLAGS.base_log_dir+FLAGS.experiment_name+'/'

def main():
    # Prepare data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
      
    #Avoid tensorboard error on IPython
    tf.reset_default_graph()
    # Create the model
    if FLAGS.arch_name == "MLP":
        model = model_example(params.MLP_model_params)
    if FLAGS.arch_name == "Deep_mnist":
        model = deep_mnist(params.MLP_model_params)
    
    #Make tf.summary for tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir+'/train',model.train_step.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir+'/test')
      
    #Start tf session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in tqdm(range(FLAGS.total_epoch)):
            with tf.variable_scope("training_steps"):
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                  
                _,summary = sess.run([model.train_step,model.merged], 
                          feed_dict={model.x: batch_xs, model.y: batch_ys,model.keep_prob: 0.5})
                train_writer.add_summary(summary, epoch)
              
            if epoch % FLAGS.eval_per_epoch == 0:  # Record summaries and test-set accuracy
                with tf.variable_scope("testing_steps"):
                    accuracy,summary = sess.run([model.accuracy,model.merged], 
                                               feed_dict={
                                                       model.x: mnist.test.images, 
                                                       model.y: mnist.test.labels,
                                                       model.keep_prob: 1.0
                                                       })
                    test_writer.add_summary(summary, epoch)
                print('accuracy at step %s: %s' % (epoch, accuracy))
            
            if epoch % FLAGS.save_per_epoch == 0:
                with tf.variable_scope("Saver_steps"):
                    tf.train.Saver().save(sess, '{}/epoch_{}'.format(FLAGS.log_dir, epoch))
    
       
    print('checkout result with "tensorboard --logdir={}"'.format(FLAGS.log_dir))
  
  
if __name__ == '__main__':
    default_hp=params.default_hyper_params
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./datasets/MNIST_data/")
    parser.add_argument('--experiment_name', type=str, default="default")
    parser.add_argument('--base_log_dir', type=str, default="./generated/logdir/")
    parser.add_argument('--arch_name', type=str, default="Deep_mnist")
    parser.add_argument('--total_epoch', type=int, default=default_hp.num_epochs)
    parser.add_argument('--eval_per_epoch', type=int, default=default_hp.eval_per_epoch)
    parser.add_argument('--save_per_epoch', type=int, default=default_hp.save_per_epoch)
    parser.add_argument('--batch_size', type=int, default=default_hp.batch_size)
    FLAGS, unparsed = parser.parse_known_args()
    prepare_params()
    main()