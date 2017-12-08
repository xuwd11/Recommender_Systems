---
nav_include: 4
title: Ensemble
notebook: 06_ensemble.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}

## Introduction

In this part, we build ensemble estimators, which make predictions based on the predictions of base estimators. The goal here is to build an ensemble estimator, of which the performance is at least as good as that of the best base estimator, which help us avoid the trouble of making choices among base estimators. Such ensemble estimator would be very helpful, especially in the cases when there is variation in terms of the performance of base estimators, or in the cases where some data required for certain base estimators are not available (e.g., values of some key predictors are missing in the content filtering models).

To avoid overfitting, we train the ensemble estimators on the cross-validation set rather than the training set we used for the training of base estimators. We drop the base estimators, of which the $R^2$ score on the cross-validation set is lower than a threshold (we choose 0.05 in this case); we call the remaining base estimators as *qualified base estimators*.

Here we tried 3 strategies to build ensemble estimators (weighted average, Ridge regression, and random forest regressor).

We use Champaign dataset (20571 reviews, 878 restaurants, 8451 users) for demo purpose.

**Note:** the fitting time we report in this part does NOT include the fitting and prediction time of base estimators.
<br><br><br>











## Ensemble of collaborative filtering models

First, we built ensemble estimators for the qualified collaborative filtering base estimators.

### Weighted average

The first strategy is to make the ensemble predictions as the weighted averages of the qualified base estimators, where we use $R^2$ score on the cross-validation set as weights.

The result is shown below.







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
      <td>Ensemble1 (weighted average)</td>
      <td>0.0</td>
      <td>0.8527</td>
      <td>1.3071</td>
      <td>0.6454</td>
      <td>0.1882</td>
    </tr>
  </tbody>
</table>



![png](06_ensemble_files/06_ensemble_6_1.png)


    


The performance of weighted average is fine.
<br><br><br>

### Ridge regression

The second strategy is to perform a Ridge regression on the predictions of qualified base estimators.

The result is shown below.







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
      <td>Ensemble1 (Ridge regression)</td>
      <td>0.004</td>
      <td>1.3268</td>
      <td>1.3026</td>
      <td>0.1413</td>
      <td>0.1937</td>
    </tr>
  </tbody>
</table>



![png](06_ensemble_files/06_ensemble_9_1.png)


    


The performance on the training set is poor. We need to point out that the training set here is not "real" training set since we train the ensemble estimator on the cross-validation set.
<br><br><br>

### Random forest

The third strategy is to train a random forest regressor on the predictions of qualified base estimators. We can use the best parameters determined by cross-validation.

The result is shown below.







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
      <td>Ensemble1 (random forest)</td>
      <td>0.215</td>
      <td>1.057</td>
      <td>1.3037</td>
      <td>0.4551</td>
      <td>0.1924</td>
    </tr>
  </tbody>
</table>



![png](06_ensemble_files/06_ensemble_12_1.png)


    


The performance on the training set seems to be better than that of Ridge regression.
<br><br><br>

## Ensemble of collaborative filtering and content filtering models






### Weighted average







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
      <td>Ensemble2 (weighted average)</td>
      <td>0.0</td>
      <td>0.9007</td>
      <td>1.1591</td>
      <td>0.6043</td>
      <td>0.3616</td>
    </tr>
  </tbody>
</table>



![png](06_ensemble_files/06_ensemble_17_1.png)


    


The performance on the test set drops slightly.
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
      <td>Ensemble2 (Ridge regression)</td>
      <td>0.01</td>
      <td>1.2721</td>
      <td>1.083</td>
      <td>0.2107</td>
      <td>0.4426</td>
    </tr>
  </tbody>
</table>



![png](06_ensemble_files/06_ensemble_20_1.png)


    


The performance on the training set is bad.
<br><br><br>

### Random forest







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
      <td>Ensemble2 (random forest)</td>
      <td>0.268</td>
      <td>1.0621</td>
      <td>1.0855</td>
      <td>0.4497</td>
      <td>0.4401</td>
    </tr>
  </tbody>
</table>



![png](06_ensemble_files/06_ensemble_23_1.png)


    


The performance is good on both training set and test set, indicating random forest regressor is suitable for building ensemble estimator in this case.