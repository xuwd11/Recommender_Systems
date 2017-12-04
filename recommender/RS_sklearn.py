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
    
    col_dummy = ['attributes.Alcohol', 'attributes.AgesAllowed', 'attributes.NoiseLevel', \
             'attributes.WiFi', 'attributes.Smoking', 'attributes.RestaurantsAttire']
    
    col_boolean = ['attributes.RestaurantsDelivery', 'attributes.DogsAllowed', 'attributes.BYOB', \
              'attributes.RestaurantsTableService', 'attributes.RestaurantsCounterService', \
              'attributes.Corkage', 'attributes.BusinessAcceptsBitcoin', 'attributes.WheelchairAccessible', \
              'attributes.BusinessAcceptsCreditCards', 'attributes.BusinessParking.lot', 'attributes.DriveThru', \
              'attributes.HasTV', 'attributes.BusinessParking.street', 'attributes.AcceptsInsurance', \
              'attributes.BusinessParking.valet', 'attributes.BYOBCorkage', 'attributes.BusinessParking.garage', \
              'attributes.ByAppointmentOnly', 'attributes.Caters', 'attributes.RestaurantsReservations', \
              'attributes.RestaurantsTakeOut', 'attributes.BikeParking', 'attributes.OutdoorSeating',\
              'attributes.BusinessParking.validated']
    
    df_ubr = df_ubr.fillna(0)
    
    col_drop = ['user_id', 'business_id', 'postal_code', 'latitude', 'categories', 'name_x', \
'neighborhood', 'review_count_x', 'state', 'address', 'hours.Sunday', 'hours.Monday','hours.Tuesday','hours.Wednesday','hours.Thursday','hours.Friday','hours.Saturday',\
'longitude', 'elite', 'friends', 'name_y', 'city']

    df_ubr = df_ubr.drop(col_drop, 1)
    df_ubr = pd.get_dummies(df_ubr, columns=col_dummy, drop_first=True)
    alter_boolean(df_ubr, col_boolean)
    df_ubr['yelping_since'] = df_ubr['yelping_since'].apply(lambda x: int(x[0:4]) - 2005.0)
    df_ubr['stars'] = df_ubr['stars'].apply(lambda x: x)
    df_ubr['attributes.Ambience.divey'] = df_ubr['attributes.Ambience.divey'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Ambience.casual'] = df_ubr['attributes.Ambience.casual'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Ambience.classy'] = df_ubr['attributes.Ambience.classy'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Ambience.hipster'] = df_ubr['attributes.Ambience.hipster'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Ambience.intimate'] = df_ubr['attributes.Ambience.intimate'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Ambience.romantic'] = df_ubr['attributes.Ambience.romantic'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Ambience.touristy'] = df_ubr['attributes.Ambience.touristy'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Ambience.trendy'] = df_ubr['attributes.Ambience.trendy'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Ambience.upscale'] = df_ubr['attributes.Ambience.upscale'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.BestNights.friday'] = df_ubr['attributes.BestNights.friday'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.BestNights.monday'] = df_ubr['attributes.BestNights.monday'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.BestNights.thursday'] = df_ubr['attributes.BestNights.thursday'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.BestNights.tuesday'] = df_ubr['attributes.BestNights.tuesday'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.BestNights.wednesday'] = df_ubr['attributes.BestNights.wednesday'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.BestNights.saturday'] = df_ubr['attributes.BestNights.saturday'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.BestNights.sunday'] = df_ubr['attributes.BestNights.sunday'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.DietaryRestrictions.dairy-free'] = df_ubr['attributes.DietaryRestrictions.dairy-free'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.DietaryRestrictions.halal'] = df_ubr['attributes.DietaryRestrictions.halal'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.DietaryRestrictions.kosher'] = df_ubr['attributes.DietaryRestrictions.kosher'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.DietaryRestrictions.soy-free'] = df_ubr['attributes.DietaryRestrictions.soy-free'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.DietaryRestrictions.vegan'] = df_ubr['attributes.DietaryRestrictions.vegan'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.DietaryRestrictions.vegetarian'] = df_ubr['attributes.DietaryRestrictions.vegetarian'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.GoodForDancing'] = df_ubr['attributes.GoodForDancing'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.GoodForKids'] = df_ubr['attributes.GoodForKids'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.GoodForMeal.lunch'] = df_ubr['attributes.GoodForMeal.lunch'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.GoodForMeal.brunch'] = df_ubr['attributes.GoodForMeal.brunch'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.GoodForMeal.dinner'] = df_ubr['attributes.GoodForMeal.dinner'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.GoodForMeal.latenight'] = df_ubr['attributes.GoodForMeal.latenight'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.RestaurantsGoodForGroups'] = df_ubr['attributes.RestaurantsGoodForGroups'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Music.background_music'] = df_ubr['attributes.Music.background_music'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Music.dj'] = df_ubr['attributes.Music.dj'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Music.jukebox'] = df_ubr['attributes.Music.jukebox'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Music.karaoke'] = df_ubr['attributes.Music.karaoke'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Music.live'] = df_ubr['attributes.Music.live'].apply(lambda x: 1 if x == True else 0)
    df_ubr['attributes.Music.video'] = df_ubr['attributes.Music.video'].apply(lambda x: 1 if x == True else 0)

    if not return_columns:
        return df_ubr.values
    else:
        return df_ubr.values, df_ubr.columns.values
    
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