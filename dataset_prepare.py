"""
A dataset prepare(download,extract,and simply pre-processing) util
Some code was borrowed from https://github.com/petewarden/tensorflow_makefile
usage: main.py [options] 
options:
    --dataset_name=<dir>        Name of dataset [default: MNIST].
    --base_url                  Root url for datasets [default: http://yann.lecun.com/exdb/mnist/].
    --save_base_dir             Which to save extracted dataset files [default: ./datasets/].
    -h, --help                  Show this help message and exit

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import argparse

import numpy as np
from scipy import ndimage
import tensorflow as tf
from six.moves import urllib
from preprocessing_util import expend_training_data

# maybe_download
def maybe_download(source_url, filename, dirpath):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(dirpath):
        tf.gfile.MakeDirs(dirpath)
    filepath = os.path.join(dirpath, filename)
    if not tf.gfile.Exists(filepath):
        print('Start downloading', filename)
        filepath, _ = urllib.request.urlretrieve(source_url + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    else :
        print('Already existed', filename)
    return filepath

def general_maybe_download(dataset_name, save_base_dir, base_url, filename_list):
    save_path=[]
    dirpath = os.path.join(save_base_dir, dataset_name)
    if not tf.gfile.Exists(dirpath):
        tf.gfile.MakeDirs(dirpath)
    
    for filename in filename_list:
        save_path.append(maybe_download(base_url, filename, dirpath))
    
    print('All files downloaded.')
    return save_path
        

# maybe_saved
def maybe_save(filepath,data):
    
    # Example of storing in hdf5 format.
    # When the data was too large to put into the memory, I suggest HDF5 format instead of npy.
    # To the need of parallel in graph data loading, I will suggest TfRecord
    # But these options is not sufficiently better than naive npy format,
    # especially when dataset is small(https://gist.github.com/rossant/7b4704e8caeb8f173084)
#        f = h5py.File('MNIST_test.h5','w')   
#        f['data'] = test_data                
#        f['labels'] = test_labels            
#        f.close()  
    
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(filepath):
        np.save(filepath, data)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully saved', filepath, size, 'bytes.')
    else :
        print('Already existed', filepath)
    return filepath

def general_maybe_saver(dataset_name, save_base_dir, train_data,train_labels,test_data,test_labels):
    dirpath = os.path.join(save_base_dir, dataset_name)
    if not tf.gfile.Exists(dirpath):
        tf.gfile.MakeDirs(dirpath)
    maybe_save(os.path.join(dirpath, 'train_data.npy'), train_data)
    maybe_save(os.path.join(dirpath, 'train_labels.npy'), train_labels)
    maybe_save(os.path.join(dirpath, 'test_data.npy'), test_data)
    maybe_save(os.path.join(dirpath, 'test_labels.npy'), test_labels)

# Extract the images into a 4D tensor and rescale values
def extract_data(filename, num_images, image_size,num_channels, pixel_depth,
                 norm_shift=False, norm_scale=True):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size * image_size * num_images * num_channels)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if norm_shift:
            data = data - (pixel_depth / 2.0)
        if norm_scale:
            data = data / pixel_depth
        data = data.reshape(num_images, image_size, image_size, num_channels)
        data = np.reshape(data, [num_images, -1])
    return data

# Extract the labels into a vector of int64 label IDs.
def extract_labels(filename, num_images, num_labels):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,num_labels))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, num_labels])
    return one_hot_encoding



# Prepare MNIST data
def download_MNIST(dataset_name, save_base_dir, base_url,
                   use_norm_shift=False, use_norm_scale=True, use_data_augmentation=False):
    
    # Params for MNIST
    image_size = 28
    num_channels = 1
    pixel_depth = 255
    num_labels = 10
    #VALIDATION_SIZE = 5000  # Size of the validation set.
    
    filename_list = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    save_list = general_maybe_download(dataset_name, save_base_dir, base_url, filename_list)

    # Extract it into numpy arrays.
    train_data = extract_data(save_list[0], 60000, image_size,num_channels, pixel_depth,
                              use_norm_shift, use_norm_scale)
    train_labels = extract_labels(save_list[1], 60000, num_labels)
    
    test_data = extract_data(save_list[2], 10000, image_size,num_channels, pixel_depth,
                             use_norm_shift, use_norm_scale)
    test_labels = extract_labels(save_list[3], 10000, num_labels)

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = np.concatenate((train_data, train_labels), axis=1)

    train_size = train_total_data.shape[0]

    general_maybe_saver(dataset_name, save_base_dir,train_data,train_labels,test_data,test_labels)
    print ('MNIST data prepared.')

def main():
    if FLAGS.dataset_name == 'MNIST':
        download_MNIST(FLAGS.dataset_name, FLAGS.save_base_dir, FLAGS.base_url,
                           use_norm_shift=False, use_norm_scale=True, use_data_augmentation=False)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--base_url', type=str, default='http://yann.lecun.com/exdb/mnist/')
    parser.add_argument('--save_base_dir', type=str, default='./datasets/')
    parser.add_argument('--filename_list', type=str, default=None)
    FLAGS, unparsed = parser.parse_known_args()
    main()