---
nav_include: 3
title: Content Filtering
notebook: 05_content_filtering.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}

## Introduction

By merging the user.csv and business.csv through the user_id and business_id in review, we get a complete X (predictor) matrix. From EDA we find that review score is strongly associated with some of the properties of users, or can be differented greatly on some attributed of restaurants. Therefore, we decide to build content filtering models, including linear regression, ridge regression, logistic regression and random forest regressor.
<br><br><br>

## Content filtering models

### Feature Selection
Before building the model, we preprocess the data by dropping certain uncorrelated variables, such as longitutde, latitude and postal code, as well as some string varibles, such as names, friends and so on. We deal with the missing values with imputation. Also, we didn't include all the over 100 variables into consideration, because some of them don't have strong and clear relationships with review score but increase the multicollinearity and sparsity of predictor matrix.

Instead, we pick 51 important predictors through EDA, which shows some relationships with review stars. For example, as shown in the EDA, review stars have various distributions on different dummy attributes of restaurant, such as different bestnight, music, food restriction, ambience and "good for" emphasis. Also, some of the categories in categorical attributes of restaurant show different patterns with others. Instead of taking dummy, we group by the different pattern and encode the categories according to their effect on review stars based on EDA. For instance, for attribute "RestaurantAttire", only "formal" appear a different pattern compared with other categories in the box plot. So we only encode it with 1 and other categories with 0.
For properties of users, we include variables that shows strong relationships with review stars, such as users'average star, fans, review count etc. 

Here we assume: 
1. average ratings of user and restaurants are always available;

2. the average ratings in the user and business tables wouldn't deviate a lot from the averages we learn from data in the training set; thus it would be valid to include average rating from 2 tables as features in our model.

### Linear regression

First of all, we perform general OLS to have a preliminary understanding of content filtering. Linear Regression result already indicates a great performance on test set compared with Baseline (Regression) model. Similar criterion values on both training and test set means including more attributes and using content filtering could fix overfitting well. In the following, we would implement certain models we learn from class to see if we could improve it.







<table  class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>model</th>
      <th>fitting time (s)</th>
      <th>train RMSE</th>
      <th>test RMSE</th>
      <th>train $R^2$</th>
      <th>test $R^2$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear regression</td>
      <td>0.028</td>
      <td>1.0772</td>
      <td>1.0973</td>
      <td>0.4341</td>
      <td>0.4279</td>
    </tr>
  </tbody>
</table>



![png](05_content_filtering_files/05_content_filtering_6_1.png)


    


<br><br><br>

### Ridge regression







<table  class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>model</th>
      <th>fitting time (s)</th>
      <th>train RMSE</th>
      <th>test RMSE</th>
      <th>train $R^2$</th>
      <th>test $R^2$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ridge regression</td>
      <td>0.064</td>
      <td>1.0772</td>
      <td>1.0973</td>
      <td>0.4341</td>
      <td>0.4279</td>
    </tr>
  </tbody>
</table>



![png](05_content_filtering_files/05_content_filtering_9_1.png)


    


<br><br><br>

### Lasso regression







<table  class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>model</th>
      <th>fitting time (s)</th>
      <th>train RMSE</th>
      <th>test RMSE</th>
      <th>train $R^2$</th>
      <th>test $R^2$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Lasso regression</td>
      <td>0.256</td>
      <td>1.0786</td>
      <td>1.0961</td>
      <td>0.4326</td>
      <td>0.4291</td>
    </tr>
  </tbody>
</table>



![png](05_content_filtering_files/05_content_filtering_12_1.png)


    


Regression with regularization doesn't seem to improve a lot from OLS. Reason might be we already select important predictors. In fact, we try to use the complete predictors matrix (over 100 predictors) and try with RidgeCV/LassoCV. Results seem to be even worse. Manually selecting predictors might be better in interpretation and prediction. So in the following we keep using the selected predictor matrix rather than keep trying dimension reduction model such as stepwise/PCA.
<br><br><br>

### Logistic regression







<table  class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>model</th>
      <th>fitting time (s)</th>
      <th>train RMSE</th>
      <th>test RMSE</th>
      <th>train $R^2$</th>
      <th>test $R^2$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Logistic regression</td>
      <td>2.6832</td>
      <td>1.3889</td>
      <td>1.4052</td>
      <td>0.0591</td>
      <td>0.0618</td>
    </tr>
  </tbody>
</table>



![png](05_content_filtering_files/05_content_filtering_15_1.png)


    


Besides regression model, we could also implement classification model. Here we try logistic regression with cross validation. Classification accuracy seem to improve compared with regression model, but RMSE get worse on both training and test set. As a matter of fact, we consider RMSE as a more important assessment because we had better not to punish at the same scale between predicting 5 as 1 and predicitng 5 as 4. This also explains why classification model doesn't work as well as regression model due to the same punishment scale in loss function. So we no longer consider other classification models such as SVM etc.
<br><br><br>

### Random forest regressor

We come back to regression model and try random forest regressor model. We choose appropriate parameters, including n_estimator and max_depth, through cross validation (GridSearchCV). Results show that this model performs better in RMSE and $R^2$ on both training and test set. Therefore, it provides us with some ideas on implementing ensemble models.












![png](05_content_filtering_files/05_content_filtering_19_0.png)













![png](05_content_filtering_files/05_content_filtering_21_0.png)








<table  class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>model</th>
      <th>fitting time (s)</th>
      <th>train RMSE</th>
      <th>test RMSE</th>
      <th>train $R^2$</th>
      <th>test $R^2$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random forest</td>
      <td>1.1121</td>
      <td>1.0265</td>
      <td>1.0858</td>
      <td>0.486</td>
      <td>0.4398</td>
    </tr>
  </tbody>
</table>



![png](05_content_filtering_files/05_content_filtering_22_1.png)


    


<br><br><br>

## Potential Drawbacks

We investigated the significance of each predictor in the linear regression model, and found something fishy.







![png](05_content_filtering_files/05_content_filtering_25_0.png)


The assumption to include average stars as features might be fishy.

1. According to the following graph of importance of coefficients, 5 out of our 51 predictors in the model are significant / important: users' average stars, restaurants' average stars, restaurants' bike parking, restaurants' food restriction is diary-free,  and years users started to use yelp. We don't feel surprise to see the first two predictors since they play a main role in all of our models. In addition, the other predictors with coefficients slightly higher than 0 are also apparent indicators of high review score according to EDA. So the most important variables in this model are users' average stars and restaurants' average stars.

2. However, a majority of users or restaurants have only sent / received one review score. Therefore, when we make prediction based on these predictors, they are not always available.

3. Also, it the amount of review stars that users our restaurants have only sent / received is small, then the known average ratings in practice might be different from the average rating in the current two tables. Therefore, it might not be reasonable to directly using the average ratings in the two tables as the average ratings features we want.