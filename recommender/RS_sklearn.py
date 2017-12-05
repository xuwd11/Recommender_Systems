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
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV

from .Baseline import BaselineMean, BaselineRegression

def alter_boolean(df_ubr, col_boolean):
    for column in col_boolean:
        df_ubr[column] = df_ubr[column].apply(lambda x: 1 if x == True else 0)
        
def get_X(X, dfb, dfu, return_columns=False):
    df_ubr = pd.merge(pd.merge(pd.DataFrame(X, columns=['user_id', 'business_id']), \
                               dfu, how='left', on='user_id'), \
                      dfb, how='left', on='business_id')
    
    df_all = pd.DataFrame()
    df_all['lasting'] = df_ubr['yelping_since'].apply(lambda x: int(x[0:4]) - 2005)
    df_all['stars'] = df_ubr['stars'].apply(lambda x: x)
    df_all['ambience_divey'] = df_ubr['attributes.Ambience.divey'].apply(lambda x: 1 if x == True else 0)
    df_all['ambience_casual'] = df_ubr['attributes.Ambience.casual'].apply(lambda x: 1 if x == True else 0)
    df_all['ambience_classy'] = df_ubr['attributes.Ambience.classy'].apply(lambda x: 1 if x == True else 0)
    df_all['ambience_hipster'] = df_ubr['attributes.Ambience.hipster'].apply(lambda x: 1 if x == True else 0)
    df_all['ambience_intimate'] = df_ubr['attributes.Ambience.intimate'].apply(lambda x: 1 if x == True else 0)
    df_all['ambience_romantic'] = df_ubr['attributes.Ambience.romantic'].apply(lambda x: 1 if x == True else 0)
    df_all['ambience_touristy'] = df_ubr['attributes.Ambience.touristy'].apply(lambda x: 1 if x == True else 0)
    df_all['ambience_trendy'] = df_ubr['attributes.Ambience.trendy'].apply(lambda x: 1 if x == True else 0)
    df_all['ambience_upscale'] = df_ubr['attributes.Ambience.upscale'].apply(lambda x: 1 if x == True else 0)
    df_all['bestnight_5'] = df_ubr['attributes.BestNights.friday'].apply(lambda x: 1 if x == True else 0)
    df_all['bestnight_1'] = df_ubr['attributes.BestNights.monday'].apply(lambda x: 1 if x == True else 0)
    df_all['bestnight_4'] = df_ubr['attributes.BestNights.thursday'].apply(lambda x: 1 if x == True else 0)
    df_all['bestnight_2'] = df_ubr['attributes.BestNights.tuesday'].apply(lambda x: 1 if x == True else 0)
    df_all['bestnight_3'] = df_ubr['attributes.BestNights.wednesday'].apply(lambda x: 1 if x == True else 0)
    df_all['bestnight_6'] = df_ubr['attributes.BestNights.saturday'].apply(lambda x: 1 if x == True else 0)
    df_all['bestnight_7'] = df_ubr['attributes.BestNights.sunday'].apply(lambda x: 1 if x == True else 0)
    df_all['diary-free'] = df_ubr['attributes.DietaryRestrictions.dairy-free'].apply(lambda x: 1 if x == True else 0)
    df_all['halal'] = df_ubr['attributes.DietaryRestrictions.halal'].apply(lambda x: 1 if x == True else 0)
    df_all['kosher'] = df_ubr['attributes.DietaryRestrictions.kosher'].apply(lambda x: 1 if x == True else 0)
    df_all['soy-free'] = df_ubr['attributes.DietaryRestrictions.soy-free'].apply(lambda x: 1 if x == True else 0)
    df_all['vegan'] = df_ubr['attributes.DietaryRestrictions.vegan'].apply(lambda x: 1 if x == True else 0)
    df_all['vegetarian'] = df_ubr['attributes.DietaryRestrictions.vegetarian'].apply(lambda x: 1 if x == True else 0)
    df_all['good_dance'] = df_ubr['attributes.GoodForDancing'].apply(lambda x: 1 if x == True else 0)
    df_all['good_kids'] = df_ubr['attributes.GoodForKids'].apply(lambda x: 1 if x == True else 0)
    df_all['good_lunch'] = df_ubr['attributes.GoodForMeal.lunch'].apply(lambda x: 1 if x == True else 0)
    df_all['good_brunch'] = df_ubr['attributes.GoodForMeal.brunch'].apply(lambda x: 1 if x == True else 0)
    df_all['good_dinner'] = df_ubr['attributes.GoodForMeal.dinner'].apply(lambda x: 1 if x == True else 0)
    df_all['good_latenight'] = df_ubr['attributes.GoodForMeal.latenight'].apply(lambda x: 1 if x == True else 0)
    df_all['good_group'] = df_ubr['attributes.RestaurantsGoodForGroups'].apply(lambda x: 1 if x == True else 0)
    df_all['background'] = df_ubr['attributes.Music.background_music'].apply(lambda x: 1 if x == True else 0)
    df_all['dj'] = df_ubr['attributes.Music.dj'].apply(lambda x: 1 if x == True else 0)
    df_all['jukebox'] = df_ubr['attributes.Music.jukebox'].apply(lambda x: 1 if x == True else 0)
    df_all['karaoke'] = df_ubr['attributes.Music.karaoke'].apply(lambda x: 1 if x == True else 0)
    df_all['live'] = df_ubr['attributes.Music.live'].apply(lambda x: 1 if x == True else 0)
    df_all['video'] = df_ubr['attributes.Music.video'].apply(lambda x: 1 if x == True else 0)

    df_all['accept_insurance'] = df_ubr['attributes.AcceptsInsurance'].apply(lambda x: 1 if x == True else 0)
    df_all['drivu_thri'] = df_ubr['attributes.DriveThru'].apply(lambda x: 1 if x == True else 0)
    df_all['bike_parking'] = df_ubr['attributes.BikeParking'].apply(lambda x: 1 if x == pd.isnull(x) else 0)

    df_all['age_alloed'] = df_ubr['attributes.AgesAllowed'].apply(lambda x: 2 if x == '18plus' else x)
    df_all['age_alloed'] = df_ubr['attributes.AgesAllowed'].apply(lambda x: 2 if x == '19plus' else x)
    df_all['age_alloed'] = df_ubr['attributes.AgesAllowed'].apply(lambda x: 1 if x == '21plus' else x)
    df_all['age_alloed'] = df_ubr['attributes.AgesAllowed'].apply(lambda x: 1 if x == 'allages' else 0)
    df_all['wifi'] = df_ubr['attributes.WiFi'].apply(lambda x: 1 if x == 'paid' else 0)
    df_all['attire'] = df_ubr['attributes.RestaurantsAttire'].apply(lambda x: 1 if x == 'formal' else 0)
    df_all['noise'] = df_ubr['attributes.NoiseLevel'].apply(lambda x: 1 if x == 'very_loud' else 0)
    df_all['price_range'] = df_ubr['attributes.RestaurantsPriceRange2'].apply(lambda x: 0 if pd.isnull(x) else 1)

    df_all['fans'] = df_ubr['fans'].apply(lambda x: x)
    df_all['review_count'] = df_ubr['review_count_y'].apply(lambda x: x)
    df_all['average_stars'] = df_ubr['average_stars'].apply(lambda x: x)
    df_all['useful'] = df_ubr['useful'].apply(lambda x: x)
    df_all['funny'] = df_ubr['funny'].apply(lambda x: x)
    df_all['cool'] = df_ubr['cool'].apply(lambda x: x)

    if not return_columns:
        return df_all.values
    else:
        return df_all.values, df_all.columns.values
    
class RS_sklearn(BaselineRegression):
    def __init__(self, estimator=None, classification=False):
        self.estimator = estimator
        self.fitted = False
        self.classification = classification
        self.time_fitting = []
        self.time_predict = []
        self.cv_r2 = None
    
    def fit(self, X, y):
        t0 = time.time()
        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)
        self.estimator.fit(X, y)
        self.fitted = True
        self.time_fitting.append(time.time() - t0)
        return self
    
    def _predict_regression(self, X):
        X = self.scaler.transform(X)
        return self.estimator.predict(X)