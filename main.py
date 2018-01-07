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
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from models.model_example import model_example
from models.deep_mnist import deep_mnist
from models.VAE.autoencoder_vae import autoencoder

import params 

FLAGS = None

def prepare_params():
    if FLAGS.experiment_name == "default":
        now=datetime.datetime.now()
        FLAGS.experiment_name=now.strftime('%Y%m%d%H%M%S')
    FLAGS.log_dir = FLAGS.base_log_dir+FLAGS.experiment_name+'_'+FLAGS.model+'/'

def batch_decorator(batch_xs,batch_ys): 
    if FLAGS.model == "autoencoder_vae":
        #add noising step
        batch_xs_target = batch_xs
        batch_xs = batch_xs * np.random.randint(2, size=batch_xs.shape)
        batch_xs += np.random.randint(2, size=batch_xs.shape)
        
        batch_xs = batch_xs
        batch_ys = batch_xs_target

    return batch_xs,batch_ys


def main():
    # Prepare data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
    batch_xs_target = batch_xs
    
      
    #Avoid tensorboard error on IPython
    tf.reset_default_graph()
    # Create the model
    if FLAGS.model == "MLP":
        hp = params.MLP_model_params
        x = tf.placeholder(tf.float32, [None, hp.input_dim])
        y = tf.placeholder(tf.float32, [None, hp.output_dim])
        model = model_example(hp,x ,y)
        
        train_feed_dict={x: batch_xs, y: batch_ys}
        test_feed_dict={x: batch_xs, y: batch_ys}
        train_fetch_list = [model.train_step,model.merged]
        test_fetch_list = [model.accuracy,model.merged]
        
    if FLAGS.model == "Deep_mnist":
        hp = params.Deep_MNIST_model_params
        
        x = tf.placeholder(tf.float32, [None, hp.input_dim])
        y = tf.placeholder(tf.float32, [None, hp.output_dim])
        keep_probe = tf.placeholder(tf.float32)
        
        model = deep_mnist(hp, x ,y, keep_probe)
        
        train_feed_dict={x: batch_xs, y: batch_ys,keep_probe: hp.keep_probe}
        test_feed_dict={x: batch_xs, y: batch_ys,keep_probe: hp.keep_probe_test}
        train_fetch_list = [model.train_step,model.merged]
        test_fetch_list = [model.accuracy,model.merged]
        
    if FLAGS.model == "autoencoder_vae":
        hp = params.autoencoder_vae_model_params
        
        x = tf.placeholder(tf.float32, [None, hp.input_dim])
        x_hat = tf.placeholder(tf.float32, [None, hp.input_dim])
        keep_probe = tf.placeholder(tf.float32)
        
        model = autoencoder(hp, x ,x_hat, keep_probe)
        
        y=x_hat
        train_feed_dict={x: batch_xs, y: batch_xs_target,keep_probe: hp.keep_probe}
        test_feed_dict={x: batch_xs, y: batch_xs_target,keep_probe: hp.keep_probe_test}
        train_fetch_list = [model.train_step,model.merged]
        test_fetch_list = [model.loss_mean,model.merged]
    
    #Prepare tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir+'/train',model.train_step.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir+'/test')
    print('checkout result with "tensorboard --logdir={}"'.format(FLAGS.log_dir))
    

    #Start tf session
    with tf.Session() as sess:
#        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
      
        for epoch in tqdm(range(FLAGS.total_epoch)):
            with tf.variable_scope("training_steps"):
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                batch_xs,batch_ys = batch_decorator(batch_xs,batch_ys)

                _,summary = sess.run(train_fetch_list, feed_dict=train_feed_dict)
                train_writer.add_summary(summary, epoch)
              
            if epoch % FLAGS.eval_per_epoch == 0:  # Record summaries and test-set accuracy
                with tf.variable_scope("testing_steps"):
                    mertics,summary = sess.run(test_fetch_list, 
                                               feed_dict=test_feed_dict)
                    test_writer.add_summary(summary, epoch)
#                print('mertics at step %s: %s' % (epoch, mertics))
            
            if epoch % FLAGS.save_per_epoch == 0:
                with tf.variable_scope("Saver_steps"):
                    tf.train.Saver().save(sess, '{}/epoch_{}'.format(FLAGS.log_dir, epoch))


  
if __name__ == '__main__':
    default_hp=params.default_hyper_params
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./datasets/MNIST_data/")
    parser.add_argument('--experiment_name', type=str, default="default")
    parser.add_argument('--base_log_dir', type=str, default="./generated/logdir/")
    parser.add_argument('--model', type=str, default="autoencoder_vae")
    parser.add_argument('--total_epoch', type=int, default=default_hp.num_epochs)
    parser.add_argument('--eval_per_epoch', type=int, default=default_hp.eval_per_epoch)
    parser.add_argument('--save_per_epoch', type=int, default=default_hp.save_per_epoch)
    parser.add_argument('--batch_size', type=int, default=default_hp.batch_size)
    FLAGS, unparsed = parser.parse_known_args()
    prepare_params()
    main()