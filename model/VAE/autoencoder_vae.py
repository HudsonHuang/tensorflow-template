# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:27:18 2018

@author: TT
"""
import tensorflow as tf

# Gateway
class autoencoder(object):
    
    # Gaussian MLP as encoder
    def gaussian_MLP_encoder(self, x, n_hidden, n_output, keep_prob):
        with tf.variable_scope("gaussian_MLP_encoder"):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)
            
            # 1st hidden layer
            w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(x, w0) + b0
            h0 = tf.nn.elu(h0)
            h0 = tf.nn.dropout(h0, keep_prob)
            
            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.tanh(h1)
            h1 = tf.nn.dropout(h1, keep_prob)
            
            # output layer
            # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
            wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
            bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
            gaussian_params = tf.matmul(h1, wo) + bo
            
            # The mean parameter is unconstrained
            mean = gaussian_params[:, :n_output]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
    
        return mean, stddev
    
    
    
    # Bernoulli MLP as decoder
    def bernoulli_MLP_decoder(self, z, n_hidden, n_output, keep_prob, reuse=False):
    
        with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)
            
            # 1st hidden layer
            w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.tanh(h0)
            h0 = tf.nn.dropout(h0, keep_prob)
            
            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.elu(h1)
            h1 = tf.nn.dropout(h1, keep_prob)
            
            # output layer-mean
            wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
            bo = tf.get_variable('bo', [n_output], initializer=b_init)
            y = tf.sigmoid(tf.matmul(h1, wo) + bo)
        
        return y
    
    def decoder(self, z, dim_img, n_hidden):
        
        y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
        
        return y
    
    def __init__(self, hp, x_hat, x, keep_prob):  
        with tf.name_scope("autoencoder_vae"): 
            
            #def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):
            
            # encoding
            mu, sigma = self.gaussian_MLP_encoder(x_hat, hp.n_hidden, hp.dim_z, keep_prob)
            
            # sampling by re-parameterization technique
            z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
            
            # decoding
            y = self.bernoulli_MLP_decoder(z, hp.n_hidden, hp.input_dim, keep_prob)
            y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
            
            # loss
            marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
            
            marginal_likelihood = tf.reduce_mean(marginal_likelihood)
            tf.summary.scalar('marginal_likelihood', marginal_likelihood)
            self.KL_divergence = tf.reduce_mean(KL_divergence)
            tf.summary.scalar('KL_divergence', self.KL_divergence)
            
#            self.neg_marginal_likelihood=-marginal_likelihood
            ELBO = marginal_likelihood - KL_divergence
            
            loss = -ELBO
            self.loss_mean = tf.reduce_mean(loss)
#            with tf.control_dependencies([tf.assert_equal(tf.is_inf(loss), y)]):
#            if tf.count_nonzero(tf.is_inf(loss))==0:
#                tf.summary.histogram('loss', loss)
#            tf.summary.scalar('loss_mean', self.loss_mean)
            
            self.train_step = tf.train.AdamOptimizer(hp.learn_rate).minimize(loss)
            #        return y, z, loss, -marginal_likelihood, KL_divergence
        
        self.merged = tf.summary.merge_all()


