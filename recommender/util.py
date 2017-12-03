from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pandas as pd
from IPython.display import display, HTML, Markdown

from .IO import IO

def norm_cm(cm):
    cm = deepcopy(cm).astype(float)
    for i in range(len(cm)):
        cm[i, :] = cm[i, :] / np.sum(cm[i, :])
    return cm  

def plot_cm(cm, title='', file_name=None):
    
    # Reference:
    # https://github.com/kevin11h/YelpDatasetChallengeDataScienceAndMachineLearningUCSD
    
    cm = norm_cm(cm)
    c = plt.pcolor(cm, edgecolors='k', linewidths=4, cmap='jet', vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('Actual target label')
    plt.xlabel('Predicted target label')
    plt.xticks(0.5 + np.arange(5), np.arange(1,6))
    plt.yticks(0.5 + np.arange(5), np.arange(1,6))
    
    def show_values(pc, fmt="%.2f", **kw):
        pc.update_scalarmappable()
        for p, value in zip(pc.get_paths(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if value >= 0.3 and value <= 0.85:
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            plt.text(x, y, fmt % value, ha="center", va="center", color=color, **kw);
    
    show_values(c)
    
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')

def reg2classification(y_reg):
    y = np.round(y_reg)
    y[y < 1] = 1
    y[y > 5] = 5
    return y.astype(int)
        
def get_predictions(es, X, save2=None):
    y_reg = es.predict(X)
    y_cla = reg2classification(y_reg)
    y = [y_reg, y_cla]
    if save2 is not None:
        IO(save2).to_pickle(y)
    return y
		
def get_single_result(es, X, y):
    y_pred = get_predictions(es, X)
    rmse = np.sqrt(mean_squared_error(y, y_pred[0]))
    r2 = r2_score(y, y_pred[0])
    accuracy = accuracy_score(y, y_pred[1])
    return [y_pred, rmse, r2, accuracy]
    
def get_results(es, X_train, y_train, X_test, y_test, X_cv=None, y_cv=None, save2=None):
    s_train = get_single_result(es, X_train, y_train)
    s_test = get_single_result(es, X_test, y_test)
    scores = [s_train[1], s_test[1], s_train[2], s_test[2], s_train[3], s_test[3]]
    y_preds = [s_train[0], s_test[0]]
    if X_cv is not None:
        s_cv = get_single_result(es, X_cv, y_cv)
        y_preds.append(s_cv[0])
    results = [y_preds, scores, es.time_fitting[-1]]
    if save2 is not None:
        IO(save2).to_pickle(results)
    return results
    
def show_results(es, model_name, X_train, y_train, X_test, y_test, results=None, \
                  print_=True, plot=True, show_cv=True, show_title=True, figname=None):
    if show_title:
        display(Markdown('### {}'.format(model_name)))
    if results is None:
        results = get_results(es, X_train, y_train, X_test, y_test)
    y_preds, scores, time_fitting = results
    if print_:
        if show_cv:
            display(Markdown('''Fitting time: {:.4f} s.  
            RMSE on training set: {:.4f}.  
            RMSE on test set: {:.4f}.  
            $R^2$ on training set: {:.4f}.  
            $R^2$ on cross-validation set: {:.4f}.  
            $R^2$ on test set: {:.4f}.  
            Classification accuracy on training set: {:.4f}.  
            Classification accuracy on test set: {:.4f}.
            '''.format(time_fitting, scores[0], scores[1], scores[2], es.cv_r2, scores[3], scores[4], scores[5])))
        else:
            display(Markdown('''Fitting time: {:.4f} s.  
            RMSE on training set: {:.4f}.  
            RMSE on test set: {:.4f}.  
            $R^2$ on training set: {:.4f}.   
            $R^2$ on test set: {:.4f}.  
            Classification accuracy on training set: {:.4f}.  
            Classification accuracy on test set: {:.4f}.
            '''.format(time_fitting, scores[0], scores[1], scores[2], scores[3], scores[4], scores[5])))
    if plot:
        plt.figure(figsize=(15, 5.5))
        plt.subplot(1, 2, 1)
        plot_cm(confusion_matrix(y_train, y_preds[0][1]), \
        model_name + '\n(RMSE on training set: {:.4f})'.format(scores[0]))
        plt.subplot(1, 2, 2)
        plot_cm(confusion_matrix(y_test, y_preds[1][1]), \
        model_name + '\n(RMSE on test set: {:.4f})'.format(scores[1]))
        if figname is not None:
            plt.savefig(figname, bbox_inches='tight');
        plt.show();
    print()

def show_summaries(model_names, results, is_successful):
    recs = []
    cols = ['model', 'fitting time (s)', 'train RMSE', 'test RMSE', 'train $R^2$', 'test $R^2$']
    for i in range(len(is_successful)):
        if is_successful[i]:
            recs.append([model_names[i]] + [results[i][2]] + results[i][1][:4])
    df = pd.DataFrame.from_records(recs, columns=cols)
    pd.set_option('precision', 4)
    display(HTML(df.to_html(index=False)))
    return df
    
def get_base_predictions(results, is_successful, datanames, thres=0):
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
    weights = np.array(weights) / np.sum(weights)
    return ys_base_train, ys_base_test, ys_base_cv, weights