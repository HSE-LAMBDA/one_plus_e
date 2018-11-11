import os
import json
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from math import ceil
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

np.random.seed(10)


# Plot
def metric_boxplot(metric, ylabel='PR_AUC', ylim=None):
    """ Boxplot one metric type for different alphas.
    
    Params
    ------
        metric : dict, {'clf_name' : [(frac, metric_value), ...]}
    """
    n = len(metric.keys())
    plt.figure(figsize=(15, 7 * ceil(n/2)))

    for i, clf_name in enumerate(metric.keys()):
        plt.subplot(ceil(n/2), 2, i + 1)
        sns.boxplot(np.array(metric[clf_name])[:, 0] * 100, np.array(metric[clf_name])[:, 1])
        plt.ylim(ylim)
        plt.xlabel("% of used anomalies")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.title(clf_name)


def plot_means(metrics_dict, ylabel="PR AUC", y_lim=(0.3, 1.0), cmap=None):
    """ 
    Plot mean values of all algorithms.
    Used nonlinear 'X' axis.
    
    Params
    ------
        metrics_dict: dict, {'clf_name': [[frac, metric_value], ...]}
            metrics_dict['clf_name'].shape == (len(fraction_of_negative_used), 2)
    """
    plt.style.use('seaborn-white')
    plt.figure(figsize=(16, 7))
    
    for i, name in enumerate(metrics_dict.keys()):
        x, y = list(map(list, zip(*metrics_dict[name])))
        if i == 0:
            x2tick = {x[j]: j/len(x) for j in range(len(x))}
            plt.xticks(np.arange(len(x))/len(x), np.array(x) * 100)
            
        ticks = [x2tick[xx] for xx in x]
        if cmap is not None:
            plt.plot(ticks, y, label=name, color=cmap[i%len(cmap)])
        else:
            plt.plot(ticks, y, label=name)
    
    plt.ylabel(ylabel)
    plt.ylim(y_lim)
    plt.xlabel("% of used anomalies")
    plt.xlim(0, 1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_means_with_percentile(metrics_dict, ylabel="PR AUC", y_lim=(0.3, 1.0), 
                               cmap=None, clf_names=None, alpha=0.2):
    """ 
    Plot mean values and errors of all algorithms.
    Used nonlinear 'X' axis.
    
    Params
    ------
        metrics_dict: dict, {'clf_name': [[frac, metric_value], ...]}
            metrics_dict['clf_name'].shape == (len(fraction_of_negative_used), 2)
        clf_names : list of str
            Names of classifiers which should plot.
        alpha : float
            Transparency of filling between percentiles.
    """
    plt.style.use('seaborn-white')
    plt.figure(figsize=(16, 7))
    is_first_clf = True
    
    for i, name in enumerate(metrics_dict.keys()):
        if name not in clf_names:
            continue
        x, y = list(map(list, zip(*metrics_dict[name][0])))
        if is_first_clf:
            x2tick = {x[j]: j/len(x) for j in range(len(x))}
            plt.xticks(np.arange(len(x))/len(x), np.array(x) * 100)
            is_first_clf = False
        
        ticks = [x2tick[xx] for xx in x]
        _, y_lower = list(map(list, zip(*metrics_dict[name][1])))
        _, y_upper = list(map(list, zip(*metrics_dict[name][2])))
        if cmap is not None:
            plt.plot(ticks, y, label=name, color=cmap[i%len(cmap)])
            plt.fill_between(ticks, y_lower, y_upper, color=cmap[i%len(cmap)], alpha=alpha)
        else:
            plt.plot(ticks, y, label=name)
            plt.fill_between(ticks, y_lower, y_upper, color=cmap[i%len(cmap)], alpha=alpha)
    
    plt.ylabel(ylabel)
    plt.ylim(y_lim)
    plt.xlabel("% of used anomalies")
    plt.xlim(0, 1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# Eval
def p_at_r(y_test, y_pred, recall_value):
    """ Eval value of Precision at Recall metric. """
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    return max(precision[recall > recall_value])


def eval_metrics(y_test, y_pred, frac, metrics_dict, descr):
    """ Eval values for all metrics and add them to metrics_dict.
    
    Params
    ------
        frac : float
            Fraction of negative samples used in train procces.
            frac = n_true_neg / n_pos
        metrics_dict : dict, {'metric_name' : {'alpha' : [(frac, metric_value), ...]} }
            Dict of dicts for all metrics.
        descr : str
            Current algo description( == 'cur_alpha').
    """
    if descr in metrics_dict['pr_aucs']:
        # append the new number to the existing array at this slot
        metrics_dict['pr_aucs'][descr].append((frac, average_precision_score(y_test, y_pred)))
        metrics_dict['roc_aucs'][descr].append((frac, roc_auc_score(y_test, y_pred)))
        for recall in metrics_dict['p_at_r'].keys():
            metrics_dict['p_at_r'][recall][descr].append((frac, p_at_r(y_test, y_pred, float(recall))))
    else:
        # create a new array in this slot
        metrics_dict['pr_aucs'][descr] = [(frac, average_precision_score(y_test, y_pred))]
        metrics_dict['roc_aucs'][descr] = [(frac, roc_auc_score(y_test, y_pred))]   
        for recall in metrics_dict['p_at_r'].keys():
            metrics_dict['p_at_r'][recall][descr] = [(frac, p_at_r(y_test, y_pred, float(recall)))]


# Save
def dump(saved_dict, folder, name):
    """ Save a 'saved_dict' to .json in 'folder' with 'name'.
    
    Params
    ------
        saved_dict : dict, {'alpha' : [(frac, metric_value), ...]} or 
                           {'metric_param' : {'alpha' : [(frac, metric_value), ...]} }
            Dict of a metric. 'metric_param' == 'recall' for precision_at_recall, for example.
    """
    dict_for_json = {k: {kk: list(vv) for kk, vv in v.items()} if isinstance(v, dict) else list(v) 
                     for k, v in saved_dict.items()}
    with open(os.path.join(folder, name + ".json"), 'w', encoding="utf-8") as file:
        # writing
        json.dump(dict_for_json, file, indent=4, ensure_ascii=False)
    del dict_for_json


def get_last_dir_n(metrics_path):
    """ Return the highest number of folders which name == number"""
    try:
        last_folder_n = max(list(map(lambda name: int(name), 
                                     filter(lambda name: os.path.isdir(os.path.join(metrics_path, name)) 
                                            and name.isdecimal(), os.listdir(metrics_path)))))
    except:
        last_folder_n = 0
        
    return last_folder_n


def create_metrics_folder(metrics_path):
    """ Create new folder for metrics in 'metrics_path' dir.
    Return
    ------
        new_folder : str
            Path to new folder.
        old_folder : str
            Path to prev folder.
    """
    last_folder_n = get_last_dir_n(metrics_path)
    new_folder = os.path.join(metrics_path, str(last_folder_n + 1))
    old_folder = os.path.join(metrics_path, str(last_folder_n))
    os.makedirs(new_folder)
    
    return new_folder, old_folder


def dump_metrics(metrics_path, metrics_dict):
    """ Save all metrics from 'metrics_dict' to .json files.
        Save in 'metrics_path' dir to new folder and delete prev folder,
        because 'metrics_dict' usualy contains old values too.
        
    Params
    ------
        metrics_dict : dict, {'metric_name' : {'alpha' : [(frac, metric_value), ...]} }
            Dict of dicts for all metrics.
    """
    folder, old_folder = create_metrics_folder(metrics_path)
    print(folder)

    for metric_name, metric_values in metrics_dict.items():
        dump(metric_values, folder, metric_name)
    
    shutil.rmtree(old_folder, ignore_errors=True)
    

# Load
def load_metrics(metrics_path):
    """
    Params
    ------
        metrics_path : str
            Path to dir with all metrics for one algo.

    Return
    ------
        metrics : dict, {'metric_name' : {'clf_name' : [(frac, metric_value), ...], ...}}
            frac == fraction of negative data used
    """
    metrics = {}
    folder = os.path.join(metrics_path, str(get_last_dir_n(metrics_path)))
    for file_name in filter(lambda name: name.endswith('.json'), os.listdir(folder)):
        with open(os.path.join(folder, file_name), 'r') as file:
            metrics[file_name[:-5]] = json.load(file)
    return metrics


def init_metrics(metrics_paths):
    """ Load all metric types from directories.
    
    Params
    ------
        metrics_paths : str or list of str
            Path to dir with all metrics for one algo.
            Or list of such paths.

    Return
    ------
        pr_aucs, roc_aucs, p_at_r
            Values of these metrics.
    """
    if not isinstance(metrics_paths, (list, np.ndarray)):
        metrics_paths = [metrics_paths]
        
    pr_aucs, roc_aucs = {}, {}
    p_at_r = defaultdict(dict)
    for metrics_path in metrics_paths:
        metrics = load_metrics(metrics_path)
        pr_aucs.update(metrics['pr_aucs'])
        roc_aucs.update(metrics['roc_aucs'])
        for k, v in metrics['p_at_r'].items():
            p_at_r[k].update(v)
    return pr_aucs, roc_aucs, p_at_r


# Metrics preparation for plot
def mean_metric(metric_arr):
    """
    Params
    ------
        metric_arr : list, shape=(n_metric_values, 2), [[frac, metric_value], ...]
            frac == fraction of negative data used
        
    Return
    ------
        metric_reduce : list, shape=(n_fractions, 2)
            Mean values of metrics by frac.
    """
    metric_map = defaultdict(list)
    for x, y in metric_arr:
        metric_map[x].append(y)
    metric_reduce = [[k, np.mean(v)] for k, v in metric_map.items()]
    metric_reduce.sort()
    
    return metric_reduce


def dublicate_metric(metric_arr, fraction_of_negative_used):
    """
    Params
    ------
        metric_arr : list, shape=(1, 2), [[frac, metric_value]]
            frac == fraction of negative data used
            
        fraction_of_negative_used : list of ints, shape=(n_fractions)
            Frac values.

    Return
    ------
        dublicated_metric : list, shape=(n_fractions, 2)
            Dublicate first value of metric_arr to all fractions.
    """
    return [[frac, metric_arr[0][1]] for frac in fraction_of_negative_used]


def create_metrics_dict_by_clf(unsupervised, supervised, ours, fraction_of_negative_used):
    """
    Params
    ------
        unsupervised: dict, {'clf_name': [[frac, metric_value]]}
            unsupervised['clf_name'].shape == (1, 2)

        supervised: dict, {'clf_name': [[frac, metric_value], ...]}
            supervised['clf_name'].shape == (n_metric_values, 2)

        ours: dict, {'clf_name': [[frac, metric_value], ...]}
            ours['clf_name'].shape == (n_metric_values, 2)
    
    Return
    ------
        metrics_dict: dict, {'clf_name': [[frac, metric_value], ...]}
            metrics_dict['clf_name'].shape == (n_fractions, 2)
    """
    metrics_dict = {k: dublicate_metric(v, fraction_of_negative_used) for k, v in unsupervised.items()}
    metrics_dict.update({k: mean_metric(v) for k, v in supervised.items()})
    metrics_dict.update({k: mean_metric(v) for k, v in ours.items()})
    
    return metrics_dict


def create_metrics_dict_by_clf_with_percentile(unsupervised, supervised, ours, 
                                               fraction_of_negative_used, lower_percentile, upper_percentile):
    """
    Params
    ------
        unsupervised: dict, {'clf_name': [[frac, metric_value]]}
            unsupervised['clf_name'].shape == (1, 2)

        supervised: dict, {'clf_name': [[frac, metric_value], ...]}
            supervised['clf_name'].shape == (n_metric_values, 2)

        ours: dict, {'clf_name': [[frac, metric_value], ...]}
            ours['clf_name'].shape == (n_metric_values, 2)
    
    Return
    ------
        metrics_dict: dict, {'clf_name': [
                    [[frac, mean_value], ...],
                    [[frac, lower_percentile_value], ...],
                    [[frac, upper_percentile_value], ...]
                ]}
            metrics_dict[clf_name].shape == (3, n_fractions, 2)
    """
    
    metrics_dict = {k: [dublicate_metric(v, fraction_of_negative_used)] * 3 for k, v in unsupervised.items()}
    metrics_dict.update({k: [mean_metric(v), *get_percentiles(v, lower_percentile, upper_percentile)] 
                         for k, v in supervised.items()})
    metrics_dict.update({k: [mean_metric(v), *get_percentiles(v, lower_percentile, upper_percentile)] 
                         for k, v in ours.items()})
    
    return metrics_dict


def get_percentiles(metric_arr, lower_percentile, upper_percentile):
    """ Eval percentiles for metric_arr. 
    
    Params
    ------
        metric_arr : list, shape=(n_metric_values, 2), [[frac, metric_value], ...]
            frac == fraction of negative data used
        lower_percentile : float in [0, 100]
        upper_percentile : float in [0, 100]
        
    Returns
    -------
        metric_lower : list, shape=(n_fractions, 2), [[frac, lower_percentile_value], ...]
        metric_upper : list, shape=(n_fractions, 2), [[frac, upper_percentile_value], ...]
    """
    metric_map = defaultdict(list)
    for x, y in metric_arr:
        metric_map[x].append(y)
    metric_lower = [[k, np.percentile(v, lower_percentile)] for k, v in metric_map.items()]
    metric_upper = [[k, np.percentile(v, upper_percentile)] for k, v in metric_map.items()]
    metric_lower.sort()
    metric_upper.sort()
    return metric_lower, metric_upper