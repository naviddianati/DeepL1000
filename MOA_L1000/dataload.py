'''
Created on Jun 22, 2021

@author: Navid Dianati
'''

import numpy as np
import pandas as pd
import os


def load_data_v1():
    '''
    Load the training and holdout data 
    '''
    
    directory = "/home/navid/data/DeepL1000/train/v1"
    
    file_targets_train = os.path.join(directory, "targets_train.hdf")
    file_targets_holdout = os.path.join(directory, "targets_holdout.hdf")
    file_features_train = os.path.join(directory, "features_train.npy")
    file_features_holdout = os.path.join(directory, "features_holdout.npy")

    X_train = np.load(file_features_train)
    X_holdout = np.load(file_features_holdout)
    
    targets_train = pd.read_hdf(file_targets_train, 'root')    
    targets_holdout = pd.read_hdf(file_targets_holdout, 'root')    
    
    w_train = pd.Series(np.ones(X_train.shape[0]), index=targets_train.index)
    
    return X_train, targets_train, w_train, X_holdout, targets_holdout

