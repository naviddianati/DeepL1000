'''
Created on Jun 22, 2021

@author: Navid Dianati
'''
import logging
from sklearn.preprocessing import QuantileTransformer
import numpy as np

logger = logging.getLogger('main.transformers')


class TransformerDummy:

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    
class TransformerRemoveMed:
    '''
    Remove gene and cell medians from each sample
    RobustScale all features
    '''
    
    def __init__(self):
        pass
    
    def fit(self, X):
        pass
        
    def transform(self, X):
        q25 = np.quantile(X, 0.25, axis=1)
        q75 = np.quantile(X, 0.75, axis=1)

        med = (q25 + q75) / 2

        # Give med a dummy dimension so its shape
        # matches that of X_train and we can subtract
        med = np.expand_dims(med, 1)

        X_transformed = np.subtract(X, med)
        
        logger.info('Applied feature transformer to data')
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class TransformerQuantileTransform:
    '''
    Quantile Transform across all samples for each of
    the 7 cell lines
    '''
    
    def __init__(self):
        self.quantile_transformers = [
            QuantileTransformer(n_quantiles=100, output_distribution="normal") for i in range(7)
        ]

    def fit(self, X):
        for i in range(7):
            self.quantile_transformers[i].fit(X[:,:, i])
#         logger.info('Fitted feature transformer')
    
    def transform(self, X):
        X_transformed = np.stack(
            [self.quantile_transformers[i].transform(X[:,:, i]) for i in range(7)],
            axis=2
        )
        
#         logger.info('Applied feature transformer to data')
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

