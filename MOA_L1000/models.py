'''
Created on Jun 24, 2021

@author: navid
'''

import logging

import keras
from keras.initializers import RandomNormal, Zeros
from keras.layers import  Lambda, Multiply, Dot, GaussianNoise, Add, Average, Subtract
from keras.layers import Dense, Dropout, Conv1D, Input, Reshape, Softmax, Flatten, Concatenate, BatchNormalization
from keras.layers.core import Activation
from keras.layers.pooling import AveragePooling1D, MaxPool1D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import SGD, Adagrad, Adadelta
from keras.regularizers import l1_l2, l1, l2

import keras.backend as K
import numpy as np
import pandas as pd

# from keras.utils import plot_model
logger = logging.getLogger()


def get_model_1(n_input, n_output):
        
    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.1)(layer1)
    layer1 = Dense(64, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.1)(layer2)
    layer2 = Dense(64, activation="elu")(layer2)
    
    layer3 = BatchNormalization()(layer2)
    layer3 = Dropout(0.1)(layer3)
    layer3 = Dense(32, activation="elu")(layer3)
    
    layer4 = BatchNormalization()(layer3)
    layer4 = Dropout(0.1)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
    layer6 = BatchNormalization()(layer5)
    layer6 = Dropout(0.1)(layer6)
    layer6 = Dense(1600, activation="elu")(layer6)
    
    layer7 = BatchNormalization()(layer6)
    layer7 = Dropout(0.1)(layer7)
    layer7 = Dense(1600, activation="elu")(layer7)
    
    layer8 = BatchNormalization()(layer7)
    layer8 = Dropout(0.1)(layer8)
    layer8 = Dense(980, activation="elu")(layer8)
    
    layer9 = BatchNormalization()(layer8)
    layer9 = Dropout(0.1)(layer8)
    layer9 = Dense(500, activation="elu")(layer9)
    
    output = Dense(n_output)(layer9)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_2(n_input, n_output):
        
    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.1)(layer1)
    layer1 = Dense(64, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.1)(layer2)
    layer2 = Dense(64, activation="elu")(layer2)
    
    layer3 = BatchNormalization()(layer2)
    layer3 = Dropout(0.1)(layer3)
    layer3 = Dense(32, activation="elu")(layer3)
    
    layer4 = BatchNormalization()(layer3)
    layer4 = Dropout(0.1)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
#     layer7 = BatchNormalization()(layer6)
#     layer7 = Dropout(0.1)(layer7)
#     layer7 = Dense(1600, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer5)
    layer8 = Dropout(0.3)(layer8)
    layer8 = Dense(980, activation="elu")(layer8)
    
    layer9 = BatchNormalization()(layer8)
    layer9 = Dropout(0.2)(layer8)
    layer9 = Dense(500, activation="elu")(layer9)
    
    output = Dense(n_output)(layer9)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_3(n_input, n_output):
    '''
    BEst recall of all models up to _6 (at least)
    Fewer neurons in beginning
    '''

    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.2)(layer1)
    layer1 = Dense(100, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.2)(layer2)
    layer2 = Dense(100, activation="elu")(layer2)
    
    layer3 = BatchNormalization()(layer2)
    layer3 = Dropout(0.2)(layer3)
    layer3 = Dense(32, activation="elu")(layer3)
    
    layer4 = BatchNormalization()(layer3)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
#     layer7 = BatchNormalization()(layer6)
#     layer7 = Dropout(0.1)(layer7)
#     layer7 = Dense(1600, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer5)
    layer8 = Dropout(0.3)(layer8)
    layer8 = Dense(980, activation="elu")(layer8)
    
    layer9 = BatchNormalization()(layer8)
    layer9 = Dropout(0.2)(layer8)
    layer9 = Dense(500, activation="elu")(layer9)
    
    output = Dense(n_output)(layer9)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_4(n_input, n_output):
    '''
    BEst so far
    one fewer layer at end
    Fewer neurons in beginning
    '''

    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.1)(layer1)
    layer1 = Dense(64, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.1)(layer2)
    layer2 = Dense(32, activation="elu")(layer2)
    
    layer3 = BatchNormalization()(layer2)
    layer3 = Dropout(0.1)(layer3)
    layer3 = Dense(10, activation="elu")(layer3)
    
    layer4 = BatchNormalization()(layer3)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
#     layer7 = BatchNormalization()(layer6)
#     layer7 = Dropout(0.1)(layer7)
#     layer7 = Dense(1600, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer5)
    layer8 = Dropout(0.3)(layer8)
    layer8 = Dense(980, activation="elu")(layer8)
#     
#     layer9 = BatchNormalization()(layer8)
#     layer9 = Dropout(0.2)(layer8)
#     layer9 = Dense(500, activation="elu")(layer9)
    
    output = Dense(n_output)(layer8)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_5(n_input, n_output):
    '''
    one fewer layer at end
    Fewer neurons in beginning
    '''

    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.1)(layer1)
    layer1 = Dense(64, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.1)(layer2)
    layer2 = Dense(32, activation="elu")(layer2)
    
    layer4 = BatchNormalization()(layer2)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
#     layer7 = BatchNormalization()(layer6)
#     layer7 = Dropout(0.1)(layer7)
#     layer7 = Dense(1600, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer5)
    layer8 = Dropout(0.3)(layer8)
    layer8 = Dense(980, activation="elu")(layer8)
#     
#     layer9 = BatchNormalization()(layer8)
#     layer9 = Dropout(0.2)(layer8)
#     layer9 = Dense(500, activation="elu")(layer9)
    
    output = Dense(n_output)(layer8)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_6(n_input, n_output):
    '''
    one fewer layer at end
    Fewer neurons in beginning
    '''

    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.2)(layer1)
    layer1 = Dense(64, activation="elu")(layer1)

    layer4 = BatchNormalization()(layer1)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
#     layer7 = BatchNormalization()(layer6)
#     layer7 = Dropout(0.1)(layer7)
#     layer7 = Dense(1600, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer5)
    layer8 = Dropout(0.3)(layer8)
    layer8 = Dense(980, activation="elu")(layer8)
#     
#     layer9 = BatchNormalization()(layer8)
#     layer9 = Dropout(0.2)(layer8)
#     layer9 = Dense(500, activation="elu")(layer9)
    
    output = Dense(n_output)(layer8)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_7(n_input, n_output):
    '''
    Seems to be doing great in  terms of recall
    Like 3 , one more layer at the end
    (BEst recall of all models up to _6 (at least)
    Fewer neurons in beginning)
    '''

    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.2)(layer1)
    layer1 = Dense(100, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.2)(layer2)
    layer2 = Dense(100, activation="elu")(layer2)
    
    layer3 = BatchNormalization()(layer2)
    layer3 = Dropout(0.2)(layer3)
    layer3 = Dense(32, activation="elu")(layer3)
    
    layer4 = BatchNormalization()(layer3)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
    layer7 = BatchNormalization()(layer5)
    layer7 = Dropout(0.1)(layer7)
    layer7 = Dense(980, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer7)
    layer8 = Dropout(0.3)(layer8)
    layer8 = Dense(980, activation="elu")(layer8)
    
    layer9 = BatchNormalization()(layer8)
    layer9 = Dropout(0.2)(layer8)
    layer9 = Dense(500, activation="elu")(layer9)
    
    output = Dense(n_output)(layer9)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_8(n_input, n_output):
    '''
    
    (Like 3 , one more layer at the end
    BEst recall of all models up to _6 (at least)
    Fewer neurons in beginning)
    '''

    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.2)(layer1)
    layer1 = Dense(100, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.2)(layer2)
    layer2 = Dense(100, activation="elu")(layer2)
    
    layer3 = BatchNormalization()(layer2)
    layer3 = Dropout(0.2)(layer3)
    layer3 = Dense(32, activation="elu")(layer3)
    
    layer4 = BatchNormalization()(layer3)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
    layer7 = BatchNormalization()(layer5)
    layer7 = Dropout(0.2)(layer7)
    layer7 = Dense(980, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer7)
    layer8 = Dropout(0.2)(layer8)
    layer8 = Dense(980, activation="elu")(layer8)
    
    layer9 = BatchNormalization()(layer8)
    layer9 = Dropout(0.2)(layer8)
    layer9 = Dense(980, activation="elu")(layer9)
    
    output = Dense(n_output)(layer9)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_9(n_input, n_output):
    '''
    skip layer added post flatten
    (Like 3 , one more layer at the end
    BEst recall of all models up to _6 (at least)
    Fewer neurons in beginning)
    '''

    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.2)(layer1)
    layer1 = Dense(100, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.2)(layer2)
    layer2 = Dense(100, activation="elu")(layer2)
    
    layer3 = BatchNormalization()(layer2)
    layer3 = Dropout(0.2)(layer3)
    layer3 = Dense(32, activation="elu")(layer3)
    
    layer4 = BatchNormalization()(layer3)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
    layer7 = BatchNormalization()(layer5)
    layer7 = Dropout(0.2)(layer7)
    layer7 = Dense(980, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer7)
    layer8 = Dropout(0.2)(layer8)
    layer8 = Dense(980, activation="elu")(layer8)
    
    layer9 = BatchNormalization()(layer8)
    layer9 = Dropout(0.2)(layer9)
    layer9 = Dense(980, activation="elu")(layer9)
    
    layer10 = Add()([layer9, layer7])
    
    output = Dense(n_output)(layer10)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_10(n_input, n_output):
    '''
    s2 kip layers added post flatten
    (Like 3 , one more layer at the end
    BEst recall of all models up to _6 (at least)
    Fewer neurons in beginning)
    '''

    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.2)(layer1)
    layer1 = Dense(100, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.2)(layer2)
    layer2 = Dense(100, activation="elu")(layer2)
    
    layer3 = BatchNormalization()(layer2)
    layer3 = Dropout(0.2)(layer3)
    layer3 = Dense(32, activation="elu")(layer3)
    
    layer4 = BatchNormalization()(layer3)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
    layer7 = BatchNormalization()(layer5)
    layer7 = Dropout(0.2)(layer7)
    layer7 = Dense(500, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer7)
    layer8 = Dropout(0.2)(layer8)
    layer8 = Dense(500, activation="elu")(layer8)
    
    layer9 = BatchNormalization()(layer8)
    layer9 = Dropout(0.2)(layer9)
    layer9 = Dense(500, activation="elu")(layer9)
    
    layer10 = Add()([layer9, layer7])
    layer10 = BatchNormalization()(layer10)
    layer10 = Dropout(0.2)(layer10)
    layer10 = Dense(500, activation="elu")(layer10)
    
    layer11 = Add()([layer10, layer8])
    
    
    output = Dense(n_output)(layer11)
    model = Model(inputs=input, outputs=output)
    return model


def get_model_11(n_input, n_output):
    '''
    like 9, but fewer neurons post-flatten
    skip layer added post flatten
    (Like 3 , one more layer at the end
    BEst recall of all models up to _6 (at least)
    Fewer neurons in beginning)
    '''

    input = Input(shape=(n_input, 7,))
    layer1 = BatchNormalization()(input)
    layer1 = Dropout(0.2)(layer1)
    layer1 = Dense(100, activation="elu")(layer1)
    
    layer2 = BatchNormalization()(layer1)
    layer2 = Dropout(0.2)(layer2)
    layer2 = Dense(100, activation="elu")(layer2)
    
    layer3 = BatchNormalization()(layer2)
    layer3 = Dropout(0.2)(layer3)
    layer3 = Dense(32, activation="elu")(layer3)
    
    layer4 = BatchNormalization()(layer3)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Dense(1, activation="elu")(layer4)
    
    layer5 = Flatten()(layer4)
    
#     layer6 = BatchNormalization()(layer5)
#     layer6 = Dropout(0.1)(layer6)
#     layer6 = Dense(1600, activation="elu")(layer6)
#     
    layer7 = BatchNormalization()(layer5)
    layer7 = Dropout(0.2)(layer7)
    layer7 = Dense(500, activation="elu")(layer7)
#     
    layer8 = BatchNormalization()(layer7)
    layer8 = Dropout(0.2)(layer8)
    layer8 = Dense(500, activation="elu")(layer8)
    
    layer9 = BatchNormalization()(layer8)
    layer9 = Dropout(0.2)(layer9)
    layer9 = Dense(500, activation="elu")(layer9)
    
    layer10 = Add()([layer9, layer7])
    
    output = Dense(n_output)(layer10)
    model = Model(inputs=input, outputs=output)
    return model
