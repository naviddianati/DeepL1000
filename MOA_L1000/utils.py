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
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold
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
                index: np.log(min(counts) + 100) / np.log(100 + c) for index, c in enumerate(counts)
        } 
    return class_weight


def get_class_weight_log_balanced2(y_train):
    '''Determine class weights for training'''
    counts = y_train.sum(0)
    class_weight = {
                index: np.log(min(counts) + 10) / np.log(10 + c) for index, c in enumerate(counts)
        } 
    return class_weight


def get_class_weight_equal(y_train):
    '''Determine class weights for training'''
    class_weight = {index: 1. for index, c in enumerate(y_train.columns)} 
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
        
        
def multilabel_stratified_kfold_by_drug(y, n_splits=None, **kwargs):
    '''split the drugs using multilabel stratified kfold
    then find the indices of rows corresponding to the drugs
    chosen for train and test.
    y is the targets matrix and has a 2-level index where
    level 0 is a drug id (pert_id) and level 1 is dose.'''
    assert n_splits
    # Each row is a unique drug, each colum is a class
    df_drug_annots = y.reset_index(level=1).iloc[:, 1:].reset_index().groupby('pert_id').first()
    cv = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
    for ind_train, ind_test in cv.split(df_drug_annots, df_drug_annots):
        perts_train = df_drug_annots.index[ind_train]
        perts_test = df_drug_annots.index[ind_test]

        # DataFrame indexed like y_train with columns ['id']
        # where id is a counting integer. 
        annots = pd.DataFrame(range(len(y)), index=y.index, columns=['id'])
        inds_train = annots.loc[perts_train]['id'].values
        inds_test = annots.loc[perts_test]['id'].values
        yield inds_train, inds_test
        
