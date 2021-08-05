'''
Created on Jun 22, 2021

@author: Navid Dianati

Save parameter dicts here with comments on the 
performance of model.
'''

import os

OUTDIR = "/home/navid/data/DeepL1000/output/"

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)


def get_params_DNN1():
    """ 
    """
    label = "DNN1"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_1",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_1",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN2():
    """ 
    """
    label = "DNN2"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_1",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_1",
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN3():
    """ 
    rapid lr decay
    """
    label = "DNN3"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_1",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_2",
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN4():
    """ 
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN4"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_2",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_2",
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
#             cv="utils.shufflesplit"
#             cv="utils.split_kfold_by_drugs"
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN5():
    """ 
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN5"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_3",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_1",
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN6():
    """ 
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN6"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_4",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_2",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
#             cv="utils.shufflesplit"
#             cv="utils.split_kfold_by_drugs"
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN7():
    """ 
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN7"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_5",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_2",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN8():
    """ 
    One fewer layer in beginning
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN8"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_6",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_2",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN9():
    """ 
    One fewer layer in beginning
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN9"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_7",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_1",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN10():
    """ 
    transformer removes median of each sample across all features
    (for every cell line)
    One fewer layer in beginning
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN10"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_8",
            Transformer="transformers.TransformerRemoveMed",
            lr_schedule="utils.lr_schedule_1",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN11():
    """ 
    transformer performs quantile transform across all 
    samples for each cell line
    One fewer layer in beginning
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN11"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_8",
            Transformer="transformers.TransformerQuantileTransform",
            lr_schedule="utils.lr_schedule_1",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN12():
    """ 
    model has one skip layer
    transformer performs quantile transform across all 
    samples for each cell line
    One fewer layer in beginning
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN12"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_9",
            Transformer="transformers.TransformerQuantileTransform",
            lr_schedule="utils.lr_schedule_1",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN13():
    """ 
    model has two skip layers
    transformer performs quantile transform across all 
    samples for each cell line
    One fewer layer in beginning
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN13"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_10",
            Transformer="transformers.TransformerQuantileTransform",
            lr_schedule="utils.lr_schedule_1",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN14():
    """ 
    model has two skip layers
    transformer performs quantile transform across all 
    samples for each cell line
    One fewer layer in beginning
    fewer neurons in beginning more dropout
    new model. fewer layers some higher dropout
    rapid lr decay
    """
    label = "DNN14"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_11",
            Transformer="transformers.TransformerQuantileTransform",
            lr_schedule="utils.lr_schedule_1",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold"
            )
    return params


def get_params_DNN15():
    """ 
    Like DNN8 but with the new strict kfold cv
    """
    label = "DNN15"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_6",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_2",  # rapid decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold_by_drug"
            )
    return params


def get_params_DNN16():
    """ 
    Like DNN15 but with slower lr decay
    slower decay improvd performance
    """
    label = "DNN16"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_6",
            Transformer="transformers.TransformerDummy",
            lr_schedule="utils.lr_schedule_3",  # slow decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold_by_drug"
            )
    return params


def get_params_DNN17():
    """ 
    Even slower decay
    Like DNN15 but with slower lr decay
    slower decay improvd performance
    """
    label = "DNN16"
    params = dict(
            smoothing_alpha=0.001,
            n_folds=5,
            SEEDS=[0, 1],
            test_size=0.2,
            learning_rate=0.001,
            early_stopping_patience=10,
            batch_size=128,
            n_epochs=100,
            data_loader="dataload.load_data_v1",
            model_loader="models.get_model_6",
            Transformer="transformers.TransformerQuantileTransform",
            lr_schedule="utils.lr_schedule_4",  # slow decay
            fcn_get_class_weights="utils.get_class_weight_log_balanced",
            label=label,
            output_directory=os.path.join(OUTDIR, label),
            cv="utils.multilabel_stratified_kfold_by_drug"
            )
    return params
