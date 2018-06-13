import os
# import h5py
# import shutil
import numpy as np
import pandas as pd

from math import ceil
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(10)


def get_train_test_data(path_to_dataset, column_names, train_size, test_size):
    """Read dataset, balance classes, normalize data and split to train and test"""
    # read dataset
    data = pd.read_csv(path_to_dataset, header=None, names=column_names)
    
    # balance classes
    min_class_size = min(data[data['label'] == 0].shape[0], data[data['label'] == 1].shape[0])
    data_0 = data[data['label'] == 0].reset_index(drop=True)
    data_1 = data[data['label'] == 1].reset_index(drop=True)
    data = pd.concat([data_0.iloc[:min_class_size], data_1.iloc[:min_class_size]], ignore_index=True)
    
    # train test split
    idx_train, idx_test = train_test_split(np.arange(data.shape[0], dtype='int32'), random_state=5,
                                           stratify=data['label'], train_size=train_size, test_size=test_size)
    idx_train = np.random.permutation(idx_train)
    idx_test = np.random.permutation(idx_test)
    X_train, X_test = data.drop('label', 1).iloc[idx_train], data.drop('label', 1).iloc[idx_test]
    y_train, y_test = data['label'].iloc[idx_train], data['label'].iloc[idx_test]
    
    # normalize with StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def get_gen_data(path_to_gen_file, X_good, n_samples, n_components=15, sigma=0.01, recalculate=False):
    """
    Params
    ------
        path_to_gen_file : str
            Full path to file where gen data stored(or will store)
        X_good : np.array
            Train data which have "0" label value.
        n_samples : int
            Number of generated samples.
        n_components : int
            The number of mixture components.
        sigma : float
            Standard deviation of the normal distribution which adds to data for fit GMM.
        recalculate : bool
            If it is False, that will be loaded old values from gen_file.
            Else these values will be calculated and saved to path_to_gen_file.
            
    Return
    ------
        X_gen : np.array, shape=(n_samples, X_good.shape[1])
            Generated data.
        w_gen : np.array, shape=(n_samples, )
            Weights of samples.
    """
    if os.path.isfile(path_to_gen_file) and not recalculate:
        # load
        gen_file = np.load(path_to_gen_file)
        X_gen = gen_file['X_gen']
        w_gen = gen_file['w_gen']
        return X_gen, w_gen
    
    # calculate GMM
    gm = GaussianMixture(n_components=n_components, n_init=4, covariance_type="full", verbose=0)
    gm.fit(X_good + np.random.normal(0, sigma, X_good.shape))
    print("BIC: ", gm.bic(X_good))
    
    # generate data
    X_gen = np.array(multivariate_normal.rvs(mean=gm.means_[0], cov=gm.covariances_[0], 
                                             size=ceil(gm.weights_[0] * n_samples)))
    for d in range(1, gm.n_components):
        X_gen=np.vstack((
            X_gen, 
            multivariate_normal.rvs(mean=gm.means_[d], cov=gm.covariances_[d], size=ceil(gm.weights_[d]*n_samples))
        ))
    X_gen = np.random.permutation(X_gen)[:n_samples]
    
    # weights ~ 1/max_proba
    probas = np.empty((gm.n_components, X_gen.shape[0]))
    for d in range(gm.n_components):
        probas[d] = multivariate_normal.pdf(
            X_gen, mean=gm.means_[d], cov=gm.covariances_[d], allow_singular=True
        )
    maxprob = np.max(probas, axis=0)
    w_gen = 1./(maxprob + 1e-2)
    w_gen = w_gen * n_samples/np.sum(w_gen)
    
    # save
    np.savez(path_to_gen_file, X_gen=X_gen, w_gen=w_gen)
    
    return X_gen, w_gen


def get_train_data_for_NN(X_pos, X_true_neg, X_pseudo_neg,
                          w_pos, w_true_neg, w_pseudo_neg, batch_size=128*3, alpha=0.1):
    """Prepare data for fitting NN. YOU SHOULD'N SHUFFLE DATA!!! Data already has shuffled."""
    max_size = max([X_pos.shape[0], X_true_neg.shape[0], X_pseudo_neg.shape[0]])
    sub_batch_size = int(batch_size/3)

    pos_idx = np.random.permutation([i%X_pos.shape[0] for i in np.arange(max_size)])
    true_neg_idx = np.random.permutation([i%X_true_neg.shape[0] for i in np.arange(max_size)])
    pseudo_neg_idx = np.random.permutation([i%X_pseudo_neg.shape[0] for i in np.arange(max_size)])

    res_X = []
    res_y = []
    res_w = []
    for i in range(0, max_size, sub_batch_size):
        cur_len = min([i + sub_batch_size, max_size]) - i
        idx = np.random.permutation(np.arange(cur_len * 3))
        
        res_X.extend(np.concatenate((
            X_pos[pos_idx[i:i + sub_batch_size]], 
            X_true_neg[true_neg_idx[i:i + sub_batch_size]],
            X_pseudo_neg[pseudo_neg_idx[i:i + sub_batch_size]]
        ), axis=0)[idx])
        res_y.extend(np.concatenate((
            np.zeros(cur_len), 
            np.ones(cur_len * 2)
        ), axis=0)[idx])
        res_w.extend(np.concatenate((
            w_pos[pos_idx[i:i + sub_batch_size]],
            w_true_neg[true_neg_idx[i:i + sub_batch_size]] * alpha,
            w_pseudo_neg[pseudo_neg_idx[i:i + sub_batch_size]] * (1 - alpha),
        ), axis=0)[idx])
        
    return np.array(res_X), np.array(res_y), np.array(res_w)