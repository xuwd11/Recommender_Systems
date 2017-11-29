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

from surprise import Dataset, Reader
from surprise import NormalPredictor, BaselineOnly, SVD, SVDpp, NMF, \
SlopeOne, CoClustering, KNNBasic, KNNWithMeans, KNNBaseline

from .Baseline import BaselineMean, BaselineRegression

class RS_surprise(BaselineRegression):
    def __init__(self, estimator=BaselineOnly(), classification=False):
        self.estimator = estimator
        self.fitted = False
        self.classification = classification
        self.time_fitting = []
        self.time_predict = []
        
    def fit(self, X, y):
        t0 = time.time()
        df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1))
        d = Dataset.load_from_df(df, Reader(rating_scale=(1, 5)))
        data = d.build_full_trainset()
        self.estimator.train(data)
        self.fitted = True
        self.time_fitting.append(time.time() - t0)
        return self
    
    def _predict_regression(self, X):
        y_pred = np.array([self.estimator.predict(_x[0], _x[1]).est for _x in X])
        return y_pred