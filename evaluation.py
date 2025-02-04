from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, recall_score, precision_score, confusion_matrix, average_precision_score

import numpy as np
import matplotlib.pyplot as plt

def Pat(y_test, y_pred, n=10):
    try:
        ind = np.argpartition(y_pred, -n)[-n:]
        return np.mean(y_test[ind])
    except Exception: return 0

def get_anomaly_metrics(y_test, y_pred, unsigned=True):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    def precision_at_recall(recall_=0.5): 
        result = precision[recall >= recall_]
        if result.shape[0] > 0: return result.min()
        return 0
    if unsigned: unsigned_fn = lambda x: max(x,1-x)
    else: unsigned_fn = lambda x: x
    return {
    "ROC_AUC": unsigned_fn(roc_auc_score(y_test, y_pred)),
    "PR_AUC": unsigned_fn(average_precision_score(y_test, y_pred)),
    "P@10": unsigned_fn(Pat(y_test, y_pred)),
    "Precision@0.8": unsigned_fn(precision_at_recall(0.8)),
    "Precision@0.9": unsigned_fn(precision_at_recall(0.9)),
    "Precision@0.95": unsigned_fn(precision_at_recall(0.95)),
    "Precision@0.99": unsigned_fn(precision_at_recall(0.99))
    }

def perfomance(y_test, y_pred, sample_weight=None):
#     print ("recall_score ",recall_score(y_test, np.round(y_pred)))
#     print ("precision_score ",precision_score(y_test, np.round(y_pred)))
    print ("f1_score ",f1_score(y_test, np.round(y_pred)))
    print ("confusion_matrix ")
    print (confusion_matrix(y_test, np.round(y_pred)))
    
    fig, axes = plt.subplots(nrows=3, figsize=(6, 15))
    
    ax = axes[0]
    ax.grid(True)
    precision, recall, _ = precision_recall_curve(y_test,y_pred)
    

    ax.step(recall, precision, color='b', alpha=0.2,
             where='post')
    ax.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title('2-class Precision-Recall curve: AP={0:0.3f}'.format(
              average_precision_score(y_test, y_pred, sample_weight=sample_weight)), fontsize=25)
    
    ax = axes[1]
    ax.grid(True)
    fpr, tpr, _ = roc_curve(y_test, y_pred, sample_weight=sample_weight)
    ax.plot(fpr, tpr)
    ax.set_title('ROC curve: roc_auc ={0:0.3f}'.format(
                 roc_auc_score(y_test, y_pred)), fontsize=25)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('FPR', fontsize=10)
    ax.set_ylabel('TPR', fontsize=10)
    
    ax = axes[2]
    bad_test = np.sum(y_test)
    good_test = len(y_test)-np.sum(y_test)
    ax.plot(sorted(y_pred[np.where( y_test == 0.)[0]], reverse=True), np.arange(good_test)/good_test*100, label = "good")
    ax.plot(sorted(y_pred[np.where( y_test == 1.)[0]]), np.arange(bad_test)/bad_test*100, label = "bad")
    ax.set_title('Predicted proba', fontsize=25)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    
    fig.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()
