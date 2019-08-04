#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import time
import numpy as np 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras import backend as K
from keras.utils import multi_gpu_model


#inputs = Input(input_size)

def Unet_2D(pretrained_weights = None, input_size = (256,256,1)):
    
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #drop3 = Dropout(0.5)(conv3)

    up4 = UpSampling2D(size = (2,2),interpolation='bilinear')(conv3)
    up4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up4)
    merge4 = concatenate([conv2,up4], axis = -1)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    up5 = UpSampling2D(size = (2,2),interpolation='bilinear')(conv4)
    up5 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up5)
    merge5 = concatenate([conv1,up5], axis = -1)
    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    output = Conv2D(1, 1, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    model = Model(inputs = inputs, outputs = output)
    
    parallel_model = multi_gpu_model(model, gpus=4)
    
    parallel_model.compile(loss='mean_squared_error', optimizer = Adam(lr = 1e-3))
    
    #parallel_model.summary()
    
    if(pretrained_weights):
        parallel_model.load_weights(pretrained_weights)
        
    return parallel_model 
    #return model

