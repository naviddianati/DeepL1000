'''
Created on Mar 3, 2021

@author: Navid Dianati
'''
import logging
from MOA_L1000 import utils
import numpy as np
import pandas as pd

logger = logging.getLogger('main.validation')


def compute_accuracy(df_solution, y_holdout):
    '''
    Compute the accuracy of predictions defined as the 
    fraction of samples for which one of the true labels
    ranks 1 in the predictions.
    '''
    assert df_solution.shape == y_holdout.shape
    ranks = df_solution.rank(1, ascending=False, method="min") 

    n_samples = df_solution.shape[0]
    n_correct = (ranks[ranks == 1].fillna(0) * y_holdout).sum(1)
    
    n_correct = (n_correct >= 1).sum()
    acc = float(n_correct) / n_samples
    return acc


def compute_crossentropy(df_s, df_g, epsilon=0.5):
    # We keep this epsilon thresholding just to be sure
    df_s = utils.clip(df_s)
    l1 = np.log(df_s, where=(df_g > epsilon)) * ((df_g > epsilon) + 0.)
    l2 = np.log((1 - df_s), where=(df_g <= epsilon)) * ((df_g <= epsilon) + 0.)

    N, M = df_g.shape
    crossentropy = -np.sum(np.sum(l1 + l2)) / N / M
    return crossentropy


def score(df_solution, df_ground, do_truncate=True):
    '''The main scoring function'''
    df_g = df_ground
    df_s = df_solution.loc[df_g.index,:]

    if do_truncate:
        df_s = utils.clip(df_s)
    assert df_g.shape == df_s.shape
    assert list(df_g.index) == list(df_s.index)
    assert list(df_g.columns) == list(df_s.columns)
    crossentropy = compute_crossentropy(df_s, df_g)
    return crossentropy


def compute_precision(df_solution, y_holdout):
    # Of all the entries that have rank 1, what fraction
    # have a corresponding 1 (as opposed to 0) in y_holdout?
    # group by moa.
    assert df_solution.shape == y_holdout.shape
    ranks = df_solution.rank(1, ascending=False, method="min")
    # number of moas for each signature. Mostly one, but occasionally more than one
    moa_counts = y_holdout.sum(1)

    # If a sig has N moas, this resets the ranks of its top N moas to 1
    # so that when we  multiply by y_holdout, any of its moas that are 
    # ranked in the top N are considered a hit
    np.putmask(ranks.values, np.less_equal(ranks, moa_counts.values[:, None]).values, 1)

    x = (
        (ranks[ranks == 1].reset_index(drop=True) * y_holdout.reset_index(drop=True))
        .unstack()
        .dropna()
        .reset_index()
    )
    
    x.columns = ['moa', 'sig_id', 'precision']
    precisions = x.groupby('moa')['precision'].mean().sort_values()
    return precisions


def compute_recall(df_solution, y_holdout):
    assert df_solution.shape == y_holdout.shape
    # Of all the entries of y_holdout that are 1, what
    # fraction have rank 1?
    # For cases were a sample has N MOAs, a hit is when one of
    # its MOAs has rank <= N.
    ranks = df_solution.rank(1, ascending=False, method="min")  

    # number of moas for each signature. Mostly one, but occasionaly more than one
    moa_counts = y_holdout.sum(1)

    # If a sig has N moas, this resets the ranks of its top N predicted moas to 1
    # so that when we  multiply by y_holdout, any of its moas that are 
    # ranked in the top N are considered a hit
    np.putmask(ranks.values, np.less_equal(ranks, moa_counts.values[:, None]).values, 1)

    x = (
            (ranks.reset_index(drop=True) * y_holdout.reset_index(drop=True))  # Note that ranks has been modified using the mask
            .unstack()
            .reset_index()
    )
    x.columns = ['moa', 'sig_id', 'rank']
    x = x[x['rank'] != 0]

    # recall defined by counting signatures
    recall = x.groupby('moa').apply(
            lambda d: 1. * len(d[d['rank'] == 1]) / len(d)
    )
    recall = recall.sort_values()
    recall.name = "recall"
    return recall
