# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:45:26 2018
borrowed from https://github.com/JeremyCCHsu/Gumbel-Softmax-VAE-in-tensorflow
"""
import os
import sys
import tensorflow as tf
        
def save(saver, sess, logdir, step):
    ''' Save a model to logdir/model.ckpt-[step] '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)


def load(saver, sess, logdir):
    '''
    Try to load model form a dir (search for the newest checkpoint)
    '''
    print('Trying to restore checkpoints from {} ...'.format(logdir),
        end="")
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print('  Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(
            ckpt.model_checkpoint_path
            .split('/')[-1]
            .split('-')[-1])
        print('  Global step: {}'.format(global_step))
        print('  Restoring...', end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        return global_step
    else:
        print('No checkpoint found')
        return None