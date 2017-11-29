from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

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
		
def print_results(es, model_name, X_train, y_train, X_test, y_test, \
                  classification=True, plot=True, figname=None):
    print(model_name)
    print('Fitting time: {} s.'.format(es.time_fitting[-1]))
    print('RMSE on training set: {}.'.format(es.score(X_train, y_train, scoring='rmse')))
    print('RMSE on test set: {}.'.format(es.score(X_test, y_test, scoring='rmse')))
    print('r2 on training set: {}.'.format(es.score(X_train, y_train)))
    print('r2 on test set: {}.'.format(es.score(X_test, y_test)))
    if classification:
        print('Classification accuracy on training set: {}.'.format(es.score(X_train, y_train, classification=True)))
        print('Classification accuracy on test set: {}.'.format(es.score(X_test, y_test, classification=True)))
        if plot:
            y_pred_train_label = es.predict(X_train, classification=True)
            y_pred_test_label = es.predict(X_test, classification=True)
            plt.figure(figsize=(15, 5.5))
            plt.subplot(1, 2, 1)
            plot_cm(confusion_matrix(y_train, y_pred_train_label), model_name + '\n(on training set)')
            plt.subplot(1, 2, 2)
            plot_cm(confusion_matrix(y_test, y_pred_test_label), model_name + '\n(on test set)')
            if figname is not None:
                plt.savefig(figname, bbox_inches='tight');
            plt.show();
            
            
            
        print()