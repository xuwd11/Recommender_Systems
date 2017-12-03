import pandas as pd
import numpy as np
import time
from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import Ridge, RidgeCV

from .Baseline import BaselineMean, BaselineRegression

class RS_ensemble(BaselineRegression):
    def __init__(self, algorithm='avg', classification=False):
        if not algorithm in ['avg', 'ridge']:
            raise NotImplementedError('Algorithm not implemented.')
        self.algorithm = algorithm
        self.fitted = False
        self.classification = classification
        self.time_fitting = []
        self.time_predict = []
        self.cv_r2 = None
        
    def _fit_avg(self, weights):
        self.weights = weights
        return self
    
    def _fit_ridge(self, ys_base, y):
        self.estimator = RidgeCV(normalize=True).fit(ys_base, y)
        return self
    
    def fit(self, ys_base=None, y=None, weights=None):
        t0 = time.time()
        if self.algorithm == 'avg':
            self._fit_avg(weights)
        elif self.algorithm == 'ridge':
            self._fit_ridge(ys_base, y)
        self.fitted = True
        self.time_fitting.append(time.time() - t0)
        return self
    
    def _predict_regression(self, ys_base):
        if self.algorithm == 'avg':
            return ys_base.dot(self.weights)
        elif self.algorithm == 'ridge':
            return self.estimator.predict(ys_base)