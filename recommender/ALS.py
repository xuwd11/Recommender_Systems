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

from scipy import sparse

from .Baseline import BaselineMean, BaselineRegression

class ALS1(BaselineRegression):
    # not updating biases in each iteration
    def __init__(self, alpha=2, alpha_als=2, rank=100, iterations=10, init_mean=0, init_std=0.1, \
                 random_state=0, classification=False):
        super().__init__(alpha, classification)
        self.alpha_als = alpha_als
        self.rank = rank
        self.iterations = iterations
        self.init_mean = init_mean
        self.init_std = init_std
        self.random_state = random_state
    
    def _fit_ALS_u2m(self):
        self.params_m = np.array([sparse.linalg.lsmr(self.params_u[self.X_u2m[self.ind_u2m[i]:self.ind_u2m[i+1], 0]], \
                                                     self.y_u2m[self.ind_u2m[i]:self.ind_u2m[i+1]], damp=self.alpha_als)[0] \
                                  for i in range(self.n_m)])
        return self
    
    def _fit_ALS_m2u(self):
        self.params_u = np.array([sparse.linalg.lsmr(self.params_m[self.X_m2u[self.ind_m2u[i]:self.ind_m2u[i+1], 1]], \
                                                     self.y_m2u[self.ind_m2u[i]:self.ind_m2u[i+1]], damp=self.alpha_als)[0] \
                                  for i in range(self.n_u)])
        return self
        
    def _fit_ALS(self, X, y):
        df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1))
        df_u2m = df.sort_values(by=[1, 0]).values
        self.X_u2m = df_u2m[:, :2].astype(int)
        self.y_u2m = df_u2m[:, 2]
        self.ind_u2m = np.concatenate((np.zeros(1, dtype=int), \
                                       np.cumsum(np.bincount(self.X_u2m[:, 1], minlength=self.n_m))))
        df_m2u = df.sort_values(by=[0, 1]).values
        self.X_m2u = df_m2u[:, :2].astype(int)
        self.y_m2u = df_m2u[:, 2]
        self.ind_m2u = np.concatenate((np.zeros(1, dtype=int), \
                                       np.cumsum(np.bincount(self.X_m2u[:, 0], minlength=self.n_u))))
        np.random.seed(self.random_state)
        self.params_u = np.random.normal(self.init_mean, self.init_std, (self.n_u, self.rank))
        for _ in range(self.iterations):
            self._fit_ALS_u2m()
            self._fit_ALS_m2u()
        return self
    
    def fit(self, X, y):
        t0 = time.time()
        X = self._fit_transform_id2index(X)
        y_base = self._fit_baseline_regression(X, y)._predict_no_missing(X)
        self._fit_ALS(X, y-y_base)
        self.fitted = True
        self.time_fitting.append(time.time() - t0)
        return self
    
    def _predict_ALS(self, X):
        return np.array([self.params_u[X[i, 0]].dot(self.params_m[X[i, 1]]) for i in range(len(X))])
    
    def _predict_regression(self, X):
        X = self._transform_id2index(X)
        y = self._predict_baseline(X)
        if self.iterations > 0:
            no_missing = np.logical_and(X[:, 0] != -1, X[:, 1] != -1)
            y[no_missing] = y[no_missing] + self._predict_ALS(X[no_missing])
        return y
		
class ALS2(BaselineRegression):
    # updating biases in each iteration
    def __init__(self, alpha=2, alpha_als=2, rank=100, iterations=10, init_mean=0, init_std=0.1, \
                 random_state=0, classification=False):
        super().__init__(alpha, classification)
        self.alpha_als = alpha_als
        self.rank = rank
        self.iterations = iterations
        self.init_mean = init_mean
        self.init_std = init_std
        self.random_state = random_state
    
    def _fit_ALS_u2m(self):
        y_u2m = self.y_u2m - self.params_base[1 + self.X_u2m[:, 0]]
        params_m = np.array([sparse.linalg.lsmr(np.concatenate((np.ones((self.ind_u2m[i+1]-self.ind_u2m[i], 1)), \
                                                                self.params_u\
                                                                [self.X_u2m[self.ind_u2m[i]:self.ind_u2m[i+1], 0]]), axis=1), \
                                                y_u2m[self.ind_u2m[i]:self.ind_u2m[i+1]], \
                                                damp=self.alpha_als)[0] for i in range(self.n_m)])
        self.params_base[1+self.n_u:] = params_m[:, 0]
        self.params_m = params_m[:, 1:]
        return self
    
    def _fit_ALS_m2u(self):
        y_m2u = self.y_m2u - self.params_base[1 + self.n_u + self.X_m2u[:, 1]]
        params_u = np.array([sparse.linalg.lsmr(np.concatenate((np.ones((self.ind_m2u[i+1]-self.ind_m2u[i], 1)), \
                                                                self.params_m\
                                                                [self.X_m2u[self.ind_m2u[i]:self.ind_m2u[i+1], 1]]), axis=1), \
                                                y_m2u[self.ind_m2u[i]:self.ind_m2u[i+1]], \
                                                damp=self.alpha_als)[0] for i in range(self.n_u)])
        self.params_base[1:1+self.n_u] = params_u[:, 0]
        self.params_u = params_u[:, 1:]
        return self
        
    def _fit_ALS(self, X, y):
        df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1))
        df_u2m = df.sort_values(by=[1, 0]).values
        self.X_u2m = df_u2m[:, :2].astype(int)
        self.y_u2m = df_u2m[:, 2]
        self.ind_u2m = np.concatenate((np.zeros(1, dtype=int), \
                                       np.cumsum(np.bincount(self.X_u2m[:, 1], minlength=self.n_m))))
        df_m2u = df.sort_values(by=[0, 1]).values
        self.X_m2u = df_m2u[:, :2].astype(int)
        self.y_m2u = df_m2u[:, 2]
        self.ind_m2u = np.concatenate((np.zeros(1, dtype=int), \
                                       np.cumsum(np.bincount(self.X_m2u[:, 0], minlength=self.n_u))))
        np.random.seed(self.random_state)
        self.params_u = np.random.normal(self.init_mean, self.init_std, (self.n_u, self.rank))
        for _ in range(self.iterations):
            self._fit_ALS_u2m()
            self._fit_ALS_m2u()
        return self
    
    def fit(self, X, y):
        t0 = time.time()
        X = self._fit_transform_id2index(X)
        self._fit_baseline_regression(X, y)
        self._fit_ALS(X, y-self.params_base[0])
        self.fitted = True
        self.time_fitting.append(time.time() - t0)
        return self
    
    def _predict_ALS(self, X):
        return np.array([self.params_u[X[i, 0]].dot(self.params_m[X[i, 1]]) for i in range(len(X))])
    
    def _predict_regression(self, X):
        X = self._transform_id2index(X)
        y = self._predict_baseline(X)
        if self.iterations > 0:
            no_missing = np.logical_and(X[:, 0] != -1, X[:, 1] != -1)
            y[no_missing] = y[no_missing] + self._predict_ALS(X[no_missing])
        return y