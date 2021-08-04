'''
Created on Jun 22, 2021

@author: Navid Dianati
'''
import logging
import os
import numpy as np
import pandas as pd
import scipy

from keras.callbacks import TerminateOnNaN, EarlyStopping, TensorBoard, LearningRateScheduler
from keras.losses import BinaryCrossentropy as BinaryCrossentropyLoss
from keras.metrics import BinaryCrossentropy as BinaryCrossentropyMetric
from keras.optimizers import Adam
from MOA_L1000 import mycallbacks, utils, validation, config, transformers, models

logger = logging.getLogger('main.trainer')

OUTDIR = config.OUTDIR


class Trainer():

    def __init__(self, seed=0, **params):
        self.seed = seed
        self.cv = eval(params.get("cv"))
        self.smoothing_alpha = params.get("smoothing_alpha")
        
        self.Transformer = eval(params.get('Transformer'))
        self.learning_rate = params.get('learning_rate')
        self.early_stopping_patience = params.get('early_stopping_patience')
        self.batch_size = params.get("batch_size")
        self.n_epochs = params.get("n_epochs")
        self.output_directory = params.get("output_directory")
        self.label = params.get("label")
        fcn_get_class_weights = params.get("fcn_get_class_weights")
        
        if fcn_get_class_weights:
            self.fcn_get_class_weights = eval(fcn_get_class_weights)
        else: 
            self.fcn_get_class_weights = None
            
        self.lr_schedule = eval(params.get("lr_schedule"))
        self.cv_metrics = {}
        
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        # Dict of {key: filepath} for all trained
        # models we saved to disk.
        self.dict_saved_model_filenames = {}
        
        assert self.learning_rate 
        assert self.early_stopping_patience 
        assert self.batch_size
        assert self.n_epochs 
        assert self.output_directory 
        assert self.label 
        assert self.lr_schedule
        assert self.cv
        assert self.smoothing_alpha is not None
        
        # Variables for applying the cv models to holdout
        self.n_models_applied = 0
        self.list_solution_arrays = []
        self.dict_holdout_metrics = {}

    def set_training_data(self, X, y, w, X_holdout, y_holdout):
        self.X = X
        self.y = y
        self.w = w
        
        self.X_holdout = X_holdout
        self.y_holdout = y_holdout
    
    def get_callbacks_pre(self):
        callbacks = [
            LearningRateScheduler(self.lr_schedule),
            EarlyStopping(
                patience=self.early_stopping_patience,
                monitor="val_{}".format(self.val_metric.name)
                ),

            # Keep track of the best performing weights,
            # and when training is done, revert model to 
            # those weights.
            mycallbacks.MyModelCheckpoint(
                export=False, monitor="val_{}".format(self.val_metric.name), mode='min',
                ),
            TensorBoard(log_dir=os.path.join(OUTDIR, "tensorboard_log", self.label)),
            TerminateOnNaN(),
            ]
        # Optional:
        # callback5 = TensorBoard(log_dir=os.path.join(os.path.expanduser(tensorboard_log_dir), os.path.split(config.get("outdir"))[1]))
        # callbacks.append(callback5)
        return callbacks

    def get_seed(self, i): 
        return i

    def train_model(self, model, X_train, X_test, y_train, y_test, w_train, w_test, class_weight=None, callbacks=None, **kwargs):
        logger.info('training model')
        batch_size = kwargs.get('batch_size', self.batch_size)
        n_epochs = kwargs.get('n_epochs', self.n_epochs)
        
        model.compile(
            loss=self.train_loss,
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=[self.val_metric],
            
            # In case we want out metrics to be weighted
            # weighted_metrics=[self.val_metric],
        )   

        # Decide whether to use class weights or sample weights
        if class_weight is None:
            sample_weight = w_train
            validation_data = (X_test, y_test, w_test)
        else:
            sample_weight = None
            validation_data = (X_test, y_test)

        if X_test is None:
            validation_data = None
        
        model.fit(
           x=X_train,
           y=y_train,
           sample_weight=sample_weight,
           class_weight=class_weight,
           validation_data=validation_data,
           batch_size=batch_size,
           epochs=n_epochs,
           callbacks=callbacks,
           verbose=1,
           )
        return model

    def validate_model(self, model, X_test, y_test, w_test):
        '''
        Apply model to X_test and compute a metric of performance. 
        '''
        # Cross-validation crossentropy
        y_pred = model.predict(X_test)
        
        # Our models output logits so we need to 
        # apply sigmoid post-hoc
        y_pred = scipy.special.expit(y_pred)
        y_pred = pd.DataFrame(
            y_pred,
            index=y_test.index,
            columns=y_test.columns,
            dtype="float64"
            )
        
        crossentropy = validation.compute_crossentropy(y_pred, y_test)
        accuracy = validation.compute_accuracy(y_pred, y_test)

        metrics = {
            'cv_crossentropy': crossentropy,
            'cv_accuracy': accuracy,
            }
        
        return metrics

    def register_saved_model(self, key, filepath):
        self.dict_saved_model_filenames[key] = filepath

    def evaluate_ensemble(self):
        '''
        Evaluate the performance of the "ensemble" of models trained
        on all folds, on the holdout data. Each model was applied to
        holdout during training and the predictions are already recorded
        in self.list_solution_arrays, so here we simply aggregate those
        predictions (mean and median) and compute prediction metrics.
        '''
        logger.info('Computing ensemble predictions on holdout. Ensemble size: {}'.format(len(self.list_solution_arrays)))
        df_solution_ensemble_mean = sum(self.list_solution_arrays) / len(self.list_solution_arrays) 
        df_solution_ensemble_mean = pd.DataFrame(
            df_solution_ensemble_mean,
            index=self.y_holdout.index,
            columns=self.y_holdout.columns
            )
        df_solution_ensemble_mean.to_hdf(
                os.path.join(self.output_directory, "./df_solution_ensemble_mean.h5"),
                "root",
                mode="w"
            )
        
        df_solution_ensemble_median = np.median(
            np.stack(self.list_solution_arrays),
            axis=0
            )
        df_solution_ensemble_median = pd.DataFrame(
            df_solution_ensemble_median,
            index=self.y_holdout.index,
            columns=self.y_holdout.columns
            )
        df_solution_ensemble_median.to_hdf(
            os.path.join(self.output_directory, "./df_solution_ensemble_median.h5"),
            "root",
            mode="w"
            )
        
        score_ensemble_mean = self.compute_holdout_score(df_solution_ensemble_mean)
        score_ensemble_median = self.compute_holdout_score(df_solution_ensemble_median)

        self.dict_holdout_metrics['ensemble_mean'] = {
                "holdout_score": score_ensemble_mean,
        }
        self.dict_holdout_metrics['ensemble_median'] = {
                "holdout_score": score_ensemble_median,
            }
        # Export holdout predictions of all models
        df_holdout_metrics = pd.DataFrame(self.dict_holdout_metrics).T.sort_index()
        df_holdout_metrics.astype('str').applymap(lambda s:s[:7]).to_csv(os.path.join(self.output_directory, "holdout_metrics.csv"), sep="\t")
    
    def compute_holdout_score(self, df_solution):
        
        df_ground = self.y_holdout
        s = validation.score(df_solution, df_ground)
        return s
    
    def apply_to_holdout(self, key, model, transformer, X_holdout, y_holdout):
        '''
        Apply a given model to the holdout data, after applying
        the corresponding transformer.
        We need y_holdout just to clone its index and columns.
        '''
        
        logger.info('Applying model {} to holdout'.format(key))
        
        # Transform the holdout features
        X_holdout_transformed = transformer.transform(X_holdout)
        
        # Compute predictions for holdout
        y_pred = model.predict(X_holdout_transformed)
        y_pred = scipy.special.expit(y_pred)
        
        df_solution = pd.DataFrame(
            y_pred,
            index=y_holdout.index,
            columns=y_holdout.columns,
            dtype="float64"
            )

        score = self.compute_holdout_score(df_solution)

        # Record the predictions for this model
        self.list_solution_arrays.append(df_solution.values)
        self.n_models_applied += 1
        logger.info('Holdout score for fold {}:   {:7.4f} '.format(key, score))
        self.dict_holdout_metrics[key] = {
                "holdout_score": score,
            }

    def run_cv(self, model_loader, n_folds=5, test_size=0.2, **kwargs): 
        '''
        Run cross-validation for a given model_loader. First split
        the data into train and validation, then apply preprocessing
        transformer, instantiate the model, train the model, 
        validate the model on validation data, then also apply
        on holdout and save the predictions (so the ensemble of
        models from all folds can be evaluated later).
        Note that here X is not a DataFrame but a 3D numpy array.
        '''
        # Function that will return our model
        model_loader = eval(model_loader)
        X, y, w = self.X, self.y, self.w,
        
        # Cross-validation loop
        splitter = self.cv(y=y, n_splits=n_folds, test_size=test_size)
        for fold_counter, (ids_train, ids_test) in enumerate(splitter):
            self.train_loss = BinaryCrossentropyLoss(
                label_smoothing=self.smoothing_alpha,
                from_logits=True
                )
            self.val_metric = BinaryCrossentropyMetric(
                from_logits=True
                )
        
            # Load callbacks
            callbacks = self.get_callbacks_pre()
            
            # Here X is a 3d numpy array whereas y is a Pandas DataFrame
            X_train, y_train, w_train = X[ids_train,:], y.iloc[ids_train,:], w.iloc[ids_train]
            X_test, y_test, w_test = X[ids_test,:], y.iloc[ids_test,:], w.iloc[ids_test]
            
            # Instantiate and fit transformer 
            logger.info('Transforming X_train and X_test')
            transformer = self.Transformer()
            X_train = transformer.fit_transform(X_train)
            X_test = transformer.transform(X_test)

            # Parameters to pass to the model constructor
            n_input = X_train.shape[1]
            n_output = y_train.shape[1]
            
            # Instantiate the model
            model = model_loader(n_input, n_output)
            logger.info('Number of inputs  : {}'.format(n_input))
            logger.info('Number of outputs : {}'.format(n_output))

            # Set class weights if provided
            if self.fcn_get_class_weights:
                class_weight = self.fcn_get_class_weights(y)
                logger.info('setting class weights')
            else:
                class_weight = None
                logger.info('No class weights. Using sample weights instead')
                
            # Train the model. During training, model is set to the best 
            # performing weights found during training by MyModelCheckpoint
            self.train_model(
                model,
                X_train, X_test, y_train, y_test, w_train, w_test , class_weight,
                callbacks=callbacks
                )
            
            # Model is already set to the best found by a callback
            # Save to file and register its path
            filepath = os.path.join(self.output_directory, "model_{}_{}.h5".format(self.seed, fold_counter))
            model.save(filepath)
            key = "{}_{}".format(self.seed, fold_counter)
            self.register_saved_model(key=key, filepath=filepath)
            
            # Validate the model on the test data and
            # record the validation metrics
            metrics = self.validate_model(model, X_test, y_test, w_test)
            self.cv_metrics[key] = metrics
            self.export_cv_metrics(label="cv")
            
            # Apply the model to holdout
            self.apply_to_holdout(key, model, transformer, self.X_holdout, self.y_holdout)
            
            utils.clear_all()
            
        # Evaluate the performance of the ensemble of models
        # trained durin cross
        self.evaluate_ensemble()
            
    def export_cv_metrics(self, label="cv"):
        '''label can be cv or fulldata'''
        assert label in ["cv", "fulldata"]
        df_metrics = pd.DataFrame(self.cv_metrics).sort_index()
        name = "{}_metrics.csv".format(label)
        outfile = os.path.join(self.output_directory, name)
        df_metrics.astype('str').applymap(lambda s:s[:7]).to_csv(outfile, sep="\t")
        logger.info("Exported {} metrics to file {}".format(label, outfile))
