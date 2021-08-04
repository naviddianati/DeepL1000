'''
Created on Jun 10, 2021

@author: Navid Dianati
'''

from collections import namedtuple
import itertools
import os

from cmapPy.pandasGEXpress import parse
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pandas.testing import assert_index_equal

import numpy as np
import pandas as pd

# List of all pert_idoses found in the siginfo
DOSES = ['0.04 uM', '1.11 uM', '0.37 uM', '3.33 uM', '0.125 uM', '10 uM']

# List of the 9 canonical cell lines 
CELL_IDS = ['HELA', 'MCF7', 'HT29', 'PC3', 'YAPC', 'HA1E', 'A375', 'A549', 'HEK293']

# Doses selected for the training data
SELECTED_DOSES = ['10 uM', '3.33 uM', '1.11 uM']

# Cell lines selected for training. The order is respected in all uses.
SELECTED_CELL_IDS = ['HELA', 'MCF7', 'HT29', 'PC3', 'YAPC', 'HA1E', 'A375']


def get_perts_with_all_top_n_cells(siginfo, dose, n_top_cells=7):
    '''
    For a given dose, find all pert_ids that have a signature in 
    each of the top N most abundant cell lines.
    '''
    sig = siginfo[
        (siginfo['cell_id'].isin(CELL_IDS)) & 
        (siginfo['pert_type'] == "trt_cp") & 
        (siginfo['pert_time'] == 24) & 
        (siginfo['pert_idose'] == dose)
    ]

    X = sig[['pert_id', 'cell_id']].copy()
    X['c'] = 1
    table = X.pivot_table(index="pert_id", columns='cell_id', values='c').fillna(0)

    top = table.sum(0).sort_values().tail(n_top_cells)
    print(top)

    # Find perts that have all of the top7
    counts = table.loc[:, top.index].sum(1)
    perts_alltop_n_cells = counts[counts == n_top_cells].index
    cells = top.index
    print(cells)
    sig_sub = sig[
        (sig['pert_id'].isin(perts_alltop_n_cells)) & 
        (sig['cell_id'].isin(cells))
    ]
    
    sig_ids = sig_sub['sig_id']
    
    return perts_alltop_n_cells, cells, sig_ids
  
    
def find_perts_with_sigs_in_top_n_cell_lines(siginfo, pertinfo, selected_doses, n_top_cells=7):
    '''
    For each of the selected doses, go through siginfo and find all perts that have signatures at the dose
    and the top N most common cell lines. Then make sure that the top 7 cell lines are the same for all doses.
    Return a siginfo subset corresponding to all relevant signatures, a set of represented perts, and
    a matrix of pert-MOA annotations (index pert_ids, columns MOA labels, values binary)
    '''
    
    DoseData = namedtuple('DoseData', ['perts', 'cell_ids', 'sig_ids'])
    list_dose_data = []
    for dose in selected_doses:
        perts, cell_ids, sig_ids = get_perts_with_all_top_n_cells(siginfo, dose=dose, n_top_cells=n_top_cells)
        list_dose_data.append(DoseData(perts=perts, cell_ids=cell_ids, sig_ids=sig_ids))
    
    assert len(set([tuple(sorted(list(dose_data.cell_ids))) for dose_data in list_dose_data])) == 1
    for i, dose_data1 in enumerate(list_dose_data):
        for j, dose_data2 in enumerate(list_dose_data):
            if i == j:
                continue
            sig_ids1 = dose_data1.sig_ids
            sig_ids2 = dose_data2.sig_ids
            assert set(list(sig_ids1)) & set(list(sig_ids2)) == set()
    
    list_sigs = [siginfo.set_index('sig_id').loc[dose_data.sig_ids,:] for dose_data in list_dose_data]
    
    # siginfo subset for all sig_ids of the selected
    # perts in the specified doses.
    sig = pd.concat(list_sigs, axis=0)
    
    # # The MOA labels: Given the pert_ids returned for the two doses, analyze the MOAs represented, and split data into train and holdout based on MOA frequencies.
    
    # combine the pert_ids found for the selected doses
    perts = set(itertools.chain(*[dose_data.perts for dose_data in list_dose_data]))
    
    print('Total number of perts found for the selected doses and cell lines: {:,}'.format(len(perts)))
    
    # limit to those pert_ids that are found in pertinfo
    perts = perts & set(list(pertinfo['pert_id']))
    
    # Slice of pertinfo for the selected pert_ids
    pertinfo_sub = pertinfo.set_index('pert_id').loc[perts,:].dropna()
    print("Total number of perts with available labels: {:,}".format(pertinfo_sub.shape[0]))
    
    # annotations matrix for all selected pert_ids. Columns are MOA labels
    # Rows are pert_ids. values are binary.
    annots = pd.DataFrame({pert_id: {moa.upper():1 for moa in row['moa'].split("|")} for pert_id, row in pertinfo_sub.iterrows()}).T.fillna(0)
    
    return sig, perts, annots


def generate_training_and_holdout_data(annots, sig, gct, n_splits=5):
    '''
    Cross-validation scheme: hold out 20% (at least one) compound from each MOA, and 
    20% (at least one) for validation using Multilabel KFold stratification.
    
    @param annots: DataFrame of pert-MOA binary annotations.
    @param sig: siginfo subset limitted only to the selected perts. For each dose, each 
    pert_id present should have exactly the same number N signatures in the SELECTED_CELL_LINES 
    @param gct: a GCToo object containing the feature vectors for each signature (e.g. level 5)
    
    @kwarg n_splits: for splitting the data into train and holdout, run stratified
    kfold with n_splits given by n_splits. Roughly equivalent to selecting 1/5 of the
    compounds in each MOA for holdout
    '''
    
    # Select MOAs that have at least n, m compounds in 
    # the train and holdout pert subsets obtained from
    # stratified multilabel kfold
    min_cps_train, min_cps_holdout = 4, 1
    
    # Split the data into train and holdout
    # Multilabel stratified KFold splitter. Get one split
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    train_index, test_index = next(mskf.split(annots.values, annots.values))
        
    # Intermediate pert-moa annotations matrices based on the splitter indices
    # Note: not all rows (perts) will make it.
    annots_holdout_tmp = annots.iloc[test_index,:]
    annots_train_tmp = annots.iloc[train_index,:]
    
    # Counts of perts in each MOA for train and holdout
    counts_holdout = annots_holdout_tmp.sum(0).sort_values()
    counts_train = annots_train_tmp.sum(0).sort_values()
    
    # Concatenate train and holdout counts
    counts = pd.concat([counts_train, counts_holdout], axis=1)
    counts.columns = ['train', 'holdout']
    
    # Impose count cutoffs
    print('Selecting moas that have at least {} compounds in train and {} compounds in holdout'.format(min_cps_train, min_cps_holdout))
    selected_moa_counts = counts[
        (counts['holdout'] >= min_cps_holdout) & 
        (counts['train'] >= min_cps_train) 
    ]
    
    # now the selected MOAs themselves
    selected_moas = selected_moa_counts.index
    
    print("Number of MOAs selected: {}".format(len(selected_moas)))
    
    # extract slice from pert-moa annot matrices for the selected moas
    annots_train = annots_train_tmp.loc[:, selected_moas]
    annots_holdout = annots_holdout_tmp.loc[:, selected_moas]
    
    annots_train.index.name = "pert_id"
    annots_holdout.index.name = "pert_id"
    
    # Assert that train and holdout annots matrices have identical columns
    # Assert that train and holdout pert_ids don't overlap
    assert_index_equal(annots_train.columns, annots_holdout.columns)
    assert len(set(list(annots_train.index)) & set(list(annots_holdout.index))) == 0
    
    # Now find the sig_ids for each of the included signatures of each pert (at the N selected doses) in each cell line 
    # Each of these is a DataFrame where columns are the N cell lines (in order), and the Multiindex consists of
    # pert_ids and pert_idoses. 
    sig_ids_table_train = (
        sig
        .reset_index()
        .set_index('pert_id')
        .loc[annots_train.index,:]
        .groupby(['pert_id', 'pert_idose'])
        .apply(lambda df: df.set_index('cell_id').loc[SELECTED_CELL_IDS, 'sig_id'])
    )
    
    sig_ids_table_holdout = (
        sig
        .reset_index()
        .set_index('pert_id')
        .loc[annots_holdout.index,:]
        .groupby(['pert_id', 'pert_idose'])
        .apply(lambda df: df.set_index('cell_id').loc[SELECTED_CELL_IDS, 'sig_id'])
    )
    
    pert_ids_and_doses_train = sig_ids_table_train.reset_index()[['pert_id', 'pert_idose']]
    pert_ids_and_doses_holdout = sig_ids_table_holdout.reset_index()[['pert_id', 'pert_idose']]
    pert_ids_and_doses_train.head()
    
    # Generate the targets matrices. annots have perts as index, but in targets, index must be
    # a (pert, dose) combination, so multiple rows per pert_id. We do this by merging annots
    # with a DataFrame with each row a (pert_id, dose) combination for train or holdout.
    targets_train = (
        pd
        .merge(
            pert_ids_and_doses_train,
            annots_train.reset_index(),
            on="pert_id"
        )
        .set_index(['pert_id', 'pert_idose'])
    )
    
    targets_holdout = (
        pd
        .merge(
            pert_ids_and_doses_holdout,
            annots_holdout.reset_index(),
            on="pert_id"
        )
        .set_index(['pert_id', 'pert_idose'])
    )
    
    assert_index_equal(targets_train.index, sig_ids_table_train.index)
    assert_index_equal(targets_holdout.index, sig_ids_table_holdout.index)
    
    print("Shape of targets_train: {}".format(targets_train.shape))
    print("Shape of targets_holdout: {}".format(targets_holdout.shape))
    
    # Now for each of train and holdout we have an N x 978 x 7 umpy array of features, and a, Nx100 dataframe of targets
    X_train = np.stack(
        [
            (
                gct
                .data_df
                .loc[:, sig_ids_table_train.loc[:, cell_id]]  # a column of sig_ids for that cell_id
                .values.T
            )
            for cell_id in SELECTED_CELL_IDS
        ],
        axis=2
    )
    
    X_holdout = np.stack(
        [
            (
                gct
                .data_df
                .loc[:, sig_ids_table_holdout.loc[:, cell_id]]  # a column of sig_ids for that cell_id
                .values.T
            )
            for cell_id in SELECTED_CELL_IDS
        ],
        axis=2
    )

    assert targets_train.shape[0] == X_train.shape[0]
    assert targets_holdout.shape[0] == X_holdout.shape[0]
    assert_index_equal(targets_train.columns, targets_holdout.columns)
    
    assert X_train.shape[1] == 978
    assert X_holdout.shape[1] == 978
    
    return X_train, X_holdout, targets_train, targets_holdout
    

def make_data_v1():
    README = '''
    Version 1 of the training data. For each of the doses "3.33uM and 10uM),
    find all perts that have signatures in the top 7 cell lines.
    Then inspect the MOAs, stratify the perts in each MOA 20-80, and find
    MOAs that according to this stratification have at least 4 perts in the
    larger group (train) and at least 1 pert in the smaller group (holdout)
    This yields 100 MOAs (exactly!) giving 3374 and 839 signatures in train
    and holdout respectively.
    Compile both features and targets for each set, make sure the datapoints
    are aligned  between features and targets, and write the 4 files to disk.
    
    For each data point, input (features) will be a stack of 7 vectors where
    each vector is the differential gene expression of a distinct cell line
    under the perturbation of the given drug at the given dose. For each 
    drug/dose combination, these 7 vectors together constitute the signature
    of the drug. Models will treat this stack as a one dimensional image with
    7 "channels). The targets will be a multiclass binary vector.
    '''
    datadir = "/home/navid/data/DeepL1000/raw/"
    outdir = "/home/navid/data/DeepL1000/train/v1"
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outfile_readme = os.path.join(outdir, "README")
    outfile_targets_train = os.path.join(outdir, "targets_train.hdf")
    outfile_targets_holdout = os.path.join(outdir, "targets_holdout.hdf")
    outfile_features_train = os.path.join(outdir, "features_train.npy")
    outfile_features_holdout = os.path.join(outdir, "features_holdout.npy")
    
    filename = os.path.join(datadir, "siginfo.txt")
    siginfo = pd.read_csv(filename, sep="\t")
    
    filename = os.path.join(datadir, "CORE_AB_pert_info_with_moa.txt")
    pertinfo = pd.read_csv(filename, sep="\t")
    
    filename = os.path.join(datadir, "modzs_lm_n127751x978.gctx")
    gct = parse.parse(filename)

    sig, perts, annots = find_perts_with_sigs_in_top_n_cell_lines(siginfo, pertinfo, selected_doses=SELECTED_DOSES, n_top_cells=7)
    X_train, X_holdout, targets_train, targets_holdout = generate_training_and_holdout_data(annots, sig, gct)
    
    with open(outfile_readme, 'w') as f:
        f.write(README)
        
    np.save(outfile_features_train, X_train)
    np.save(outfile_features_holdout, X_holdout)
    
    targets_train.to_hdf(outfile_targets_train, 'root')
    targets_holdout.to_hdf(outfile_targets_holdout, 'root')

