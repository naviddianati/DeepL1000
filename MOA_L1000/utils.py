'''
Created on Jun 22, 2021

@author: navid
'''

import gc
import json
import logging
import os
import random
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import keras.backend as K

logger = logging.getLogger('main.utils')


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
                

def export_params(params):
    '''
    Export the parameters of a training instance
    to file.
    '''
    output_directory = params.get('output_directory')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    outfile = os.path.join(output_directory, "params.txt")
    with open(outfile, "w") as f:
        json.dump(params, f, indent=4)


def clear_all():
    gc.collect()
    K.clear_session()


def get_class_weight_log_balanced(y_train):
    '''Determine class weights for training'''
    counts = y_train.sum(0)
    class_weight = {
                index : np.log(min(counts) + 100) / np.log(100 + c) for index, c in enumerate(counts)
        } 
    return class_weight


def get_class_weight_log_balanced2(y_train):
    '''Determine class weights for training'''
    counts = y_train.sum(0)
    class_weight = {
                index : np.log(min(counts) + 10) / np.log(10 + c) for index, c in enumerate(counts)
        } 
    return class_weight


def get_class_weight_equal(y_train):
    '''Determine class weights for training'''
    class_weight = {index : 1. for index, c in enumerate(y_train.columns)} 
    return class_weight


def lr_schedule_1(epoch, lr):
    # Learning rate schecule
    # After epoch 10, halve lr 
    # every 5 epochs
    logger.info("Epoch {}, learning rate: {}".format(epoch, lr))
    if epoch < 10:
        return lr 
    elif epoch == 10:
        return lr / 2
    else:
        if epoch % 5 == 0:
            return lr * 0.97
        else:
            return lr


def lr_schedule_2(epoch, lr):
    logger.info("Epoch {}, learning rate: {}".format(epoch, lr))
    if epoch < 10:
        return lr * 0.8 
    else:
        if epoch % 5 == 0:
            return lr * 0.95
        else:
            return lr


def lr_schedule_3(epoch, lr):
    logger.info("Epoch {}, learning rate: {}".format(epoch, lr))
    if epoch < 10:
        return lr * 0.95
    else:
        if epoch % 5 == 0:
            return lr * 0.95
        else:
            return lr


def clip(df):
    return np.clip(df.astype(np.float64), 1 - 1e-15, 1e-15)


def multilabel_stratified_kfold(y, n_splits=None, **kwargs):
    '''
    Returns integer indexes
    '''
    assert n_splits
    y = (y > 0.5) + 0
    cv = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
    return cv.split(y, y)

