# -*- coding: utf-8 -*-
import numpy as np

def autoencoder_vae_add_noise(batch_xs,batch_ys): 
    #add noising step
    batch_xs_target = batch_xs
    batch_xs = batch_xs * np.random.randint(2, size=batch_xs.shape)
    batch_xs += np.random.randint(2, size=batch_xs.shape)
    
    batch_xs = batch_xs
    batch_ys = batch_xs_target

    return batch_xs,batch_ys