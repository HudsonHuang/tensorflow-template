# -*- coding: utf-8 -*-
#/usr/bin/python2

# path
## local
data_path_base = './datasets'
logdir_path = './logdir'

# hyper_params is about experiment
class default_hyper_params:
    num_epochs = 1000
    eval_per_epoch = num_epochs//10 +1 # 10 times for eval in total train
    save_per_epoch = num_epochs//2 +1  # floor_division + 1 == ceiling_division
    batch_size = 32
    keep_probe = 0.5
    keep_probe_test = 1.0
    learn_rate = 0.001

# model_params is about model structure
class MLP_model_params(default_hyper_params):
    # data
    input_dim = 784
    output_dim = 10


    # model
    hidden_units = [256, 128,64,10]  # alias = E
    num_banks = 16
    keep_probe = 0.2

    # train
    batch_size = 32
    learn_rate = 0.001

class Deep_MNIST_model_params(default_hyper_params):
    # data
    input_dim = 784
    output_dim = 10  
    
    # train
    batch_size = 32
    learn_rate = 0.001

class autoencoder_vae_model_params(default_hyper_params):
    # data
    input_dim = 784
    
    learn_rate = 0.0001
    n_hidden = 500
    dim_z = 20
    pass

class Test1_params:
    # path
    data_path = '{}/timit/TIMIT/TEST/*/*/*.wav'.format(data_path_base)

    # test
    batch_size = 32


class Convert_params:
    # path
    data_path = '{}/arctic/bdl/*.wav'.format(data_path_base)

    # convert
    batch_size = 2
    emphasis_magnitude = 1.2