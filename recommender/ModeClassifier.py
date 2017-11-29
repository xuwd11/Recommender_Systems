import numpy as np
import time

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

class ModeClassifier:
    def __init__(self, mode=5, classification=None):
        self.mode = mode
        self.fitted = True
        self.time_fitting = []
        self.classification = classification
        
    def fit(self, X, y):
        t0 = time.time()
        self.time_fitting.append(time.time() - t0)
        return self
    
    def predict(self, X, classification=None):
        return self.mode * np.ones(len(X))
    
    def score(self, X, y, classification=None, scoring='r2'):
        if classification is None:
            classification = self.classification
        if not classification:
            if scoring == 'r2':
                return r2_score(y, self.predict(X, classification))
            elif scoring == 'mse':
                return mean_squared_error(y, self.predict(X, classification))
            elif scoring == 'rmse':
                return np.sqrt(mean_squared_error(y, self.predict(X, classification)))
            else:
                raise NotImplementedError('`scoring` should be either "r2", "mse", or "rmse".')
        else:
            return accuracy_score(y, self.predict(X, classification))