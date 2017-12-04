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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV

from .Baseline import BaselineMean, BaselineRegression
from .IO import IO

def get_base_predictions(results, is_successful, datanames, normalize=True, thres=0):
    ys_base_train = []
    ys_base_test = []
    ys_base_cv = []
    weights = []
    for i in range(len(is_successful)):
        if not is_successful[i]:
            continue
        model = IO(datanames[i]).read_pickle()
        if model.cv_r2 <= thres:
            continue
        weights.append(model.cv_r2)
        del model
        ys_base_train.append(results[i][0][0][0])
        ys_base_test.append(results[i][0][1][0])
        ys_base_cv.append(results[i][0][2][0])
    ys_base_train = np.array(ys_base_train).transpose()
    ys_base_test = np.array(ys_base_test).transpose()
    ys_base_cv = np.array(ys_base_cv).transpose()
    weights = np.array(weights)
    if normalize:
        weights = weights / np.sum(weights)
    return ys_base_train, ys_base_test, ys_base_cv, weights
    
def get_multi_base_predictions(results, is_successful, datanames, thres=0):
    ys_base_train = []
    ys_base_test = []
    ys_base_cv = []
    weights = []
    for i in range(len(is_successful)):
        y = get_base_predictions(results[i], is_successful[i], datanames[i], normalize=False, thres=thres)
        ys_base_train.append(y[0])
        ys_base_test.append(y[1])
        ys_base_cv.append(y[2])
        weights.append(y[3])
    ys_base_train = np.concatenate(ys_base_train, axis=1)
    ys_base_test = np.concatenate(ys_base_test, axis=1)
    ys_base_cv = np.concatenate(ys_base_cv, axis=1)
    weights = np.concatenate(weights)
    weights = weights / np.sum(weights)
    return ys_base_train, ys_base_test, ys_base_cv, weights

class RS_ensemble(BaselineRegression):
    def __init__(self, estimator=None, classification=False):
        self.estimator = estimator
        self.fitted = False
        self.classification = classification
        self.time_fitting = []
        self.time_predict = []
        self.cv_r2 = None
        
    def _fit_avg(self, weights):
        self.weights = weights
        return self
    
    def _fit_estimator(self, ys_base, y):
        self.estimator.fit(ys_base, y)
        return self
    
    def fit(self, ys_base=None, y=None, weights=None):
        t0 = time.time()
        if self.estimator is None:
            self._fit_avg(weights)
        else:
            self._fit_estimator(ys_base, y)
        self.fitted = True
        self.time_fitting.append(time.time() - t0)
        return self
    
    def _predict_regression(self, ys_base):
        if self.estimator is None:
            return ys_base.dot(self.weights)
        else:
            return self.estimator.predict(ys_base)