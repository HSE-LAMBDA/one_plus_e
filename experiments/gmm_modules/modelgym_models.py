import pickle
import timeit
import numpy as np
import xgboost as xgb

from hyperopt import hp
from hyperopt.pyll.base import scope
from modelgym.models import Model, LearningTask
from modelgym.utils import XYCDataset
from modelgym.models import XGBClassifier

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class IsolationForestClassifier(Model):
    def __init__(self, params=None, verbose=True):
        """
        Args:
            params (dict): parameters for model.
        """
        if params is None:
            params = {}

        self.params = {
            'n_estimators': 10,
            'max_samples': 0.1, 
            'contamination': 0.007,
            'verbose': 0,
        }
        self.params.update(params)
            
        self.model = None
        self.verbose = verbose
        
    def _set_model(self, model):
        """
        sets new model, internal method, do not use
        Args:
            model: internal model
        """
        self.model = model

    def _convert_to_dataset(self, data, label=None, cat_cols=None):
        return XYCDataset(data, label, cat_cols)

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        if self.verbose:
            # log info
            print('*****************')
            print(self.params)
        
        # clean train set
        idx_pos = np.where(dataset.y==1)[0]
        idx_neg = np.random.permutation(np.where(dataset.y!=1)[0])
        end_ind = int(self.params['contamination'] * idx_neg.shape[0])
        idx = np.concatenate([idx_pos, idx_neg[:end_ind]], axis=0)
        dtrain = self._convert_to_dataset(dataset.X[idx], dataset.y[idx])
        
        # fit
        start_time = timeit.default_timer()
        self.model = IsolationForest(
            random_state=self.params.get('random_state', 5), **self.params
        ).fit(dtrain.X, dtrain.y)
        run_time = timeit.default_timer() - start_time
        
        if self.verbose:
            # log info
            print('Fitted')
            print("Run time: {:.2f} sec".format(run_time))
            print('*****************\n')
        return self

    def save_snapshot(self, filename):
        """
        Return:
            serializable internal model state snapshot.
        """
        assert self.model is not None, "model is not fitted"
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)

        new_model = IsolationForestClassifier(model.get_params())
        new_model._set_model(model)
        return new_model

    def predict(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        return self.model.predict(dataset.X)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def predict_proba(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        assert self.is_possible_predict_proba(), "Model cannot predict probability distribution"
        return self.model.decision_function(dataset.X)

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """

        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 100, 2000, 200)),
            'max_samples': hp.uniform('max_samples', 0.1, 1), 
            'contamination': hp.loguniform('contamination', -5, 0),
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.CLASSIFICATION


class SVMClassifier(Model):
    def __init__(self, params=None, verbose=True):
        """
        Args:
            params (dict): parameters for model.
        """

        if params is None:
            params = {}

        self.params = params
        self.model = None
        self.verbose = verbose

    def _set_model(self, model):
        """
        sets new model, internal method, do not use
        Args:
            model: internal model
        """
        self.model = model

    def _convert_to_dataset(self, data, label=None, cat_cols=None):
        return XYCDataset(data, label, cat_cols)

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        cur_params = self.params.copy()
        cur_params.update(self.params['kernel'])
        
        if self.verbose:
            # log info
            print('*****************')
            print(self.params)
            print(cur_params)
        
        # clean train set
        idx = np.where(dataset.y==1)[0]
        dtrain = self._convert_to_dataset(dataset.X[idx], dataset.y[idx])
        
        # fit
        start_time = timeit.default_timer()
        self.model = OneClassSVM(
            random_state=cur_params.get('random_state', 5), **cur_params
        ).fit(dtrain.X, dtrain.y)
        run_time = timeit.default_timer() - start_time
        
        if self.verbose:
            # log info
            print('Fitted')
            print("Run time: {:.2f} sec".format(run_time))
            print('*****************\n')
        return self

    def save_snapshot(self, filename):
        """
        Return:
            serializable internal model state snapshot.
        """
        assert self.model is not None, "model is not fitted"
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)

        new_model = SVMClassifier(model.get_params())
        new_model._set_model(model)
        return new_model

    def predict(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        return self.model.predict(dataset.X)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def predict_proba(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        assert self.is_possible_predict_proba(), "Model cannot predict probability distribution"
        return self.model.decision_function(dataset.X)

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """
        space = {
            'nu': hp.choice('nu', np.arange(0.2, 0.8, 0.1)),
            'shrinking': hp.choice('shrinking', [True, False]),
            'kernel': hp.choice('kernel', [
                {
                    'kernel': 'rbf',
                    'gamma': hp.loguniform('gamma_1', -6, -1),
                },
                {
                    'kernel': 'linear',
                },
                {
                    'kernel': 'poly',
                    'degree': hp.choice('degree', range(2, 7)),
                    'gamma': hp.loguniform('gamma_3', -6, -1),
                    'coef0': hp.loguniform('coef0_3', -6, -1),
                },
                {
                    'kernel': 'sigmoid',
                    'gamma': hp.loguniform('gamma_4', -6, -1),
                    'coef0': hp.loguniform('coef0_4', -6, -1),
                }
                ])
        }
        
        return space

    @staticmethod
    def get_learning_task():
        return LearningTask.CLASSIFICATION


class XGBClassifier(Model):
    def __init__(self, params=None, verbose=True):
        """
        Args:
            params (dict): parameters for model.
        """

        if params is None:
            params = {}

        objective = 'binary:logistic'
        metric = 'logloss'
        if params.get('num_class', 2) > 2:
            # change default objective
            objective = 'multi:softprob'
            metric = 'mlogloss'

        self.params = {'objective': objective, 'eval_metric': metric,
                       'silent': 1}

        self.params.update(params)
        self.n_estimators = self.params.pop('n_estimators', 1)
        self.model = None
        self.verbose = verbose

    def _set_model(self, model):
        """
        sets new model, internal method, do not use
        Args:
            model: internal model
        """
        self.model = model

    def _convert_to_dataset(self, data, label, cat_cols=None):
        return xgb.DMatrix(data, label)

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        if self.verbose:
            # log info
            print('*****************')
            print(self.params)
        
        # fit
        start_time = timeit.default_timer()
        dtrain = self._convert_to_dataset(dataset.X, dataset.y)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators, verbose_eval=False)
        run_time = timeit.default_timer() - start_time
        
        if self.verbose:
            # log info
            print('Fitted')
            print("Run time: {:.2f} sec".format(run_time))
            print('*****************\n')
        return self

    def save_snapshot(self, filename):
        """
        Return:
            serializable internal model state snapshot.
        """
        assert self.model, "model is not fitted"
        return self.model.save_model(filename)

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        booster = xgb.Booster()
        booster.load_model(filename)
        new_model = XGBClassifier()  # idk how to pass paarameters yet
        new_model._set_model(booster)
        return new_model

    def predict(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        xgb_dataset = xgb.DMatrix(dataset.X)
        if self.params['objective'] == 'multi:softprob':
            return self.model.predict(xgb_dataset).astype(int)
        prediction = np.round(self.model.predict(xgb_dataset)).astype(int)
        if self.params.get('num_class', 2) == 2:
            return prediction
        return np.argmax(prediction, axis=-1)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        if self.params['objective'] == 'multi:softprob':
            return False
        return True

    def predict_proba(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        xgb_dataset = xgb.DMatrix(dataset.X)
        assert self.is_possible_predict_proba(), "Model cannot predict probability distribution"
        return self.model.predict(xgb_dataset)

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """

        return {
            'eta':               hp.loguniform('eta', -7, 0),
            'max_depth':         scope.int(hp.quniform('max_depth', 2, 10, 1)),
            'n_estimators':      scope.int(hp.quniform('n_estimators', 100, 1000, 100)),
            'subsample':         hp.uniform('subsample', 0.5, 1),
            'colsample_bytree':  hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight':  hp.loguniform('min_child_weight', -16, 5),
            'gamma':             hp.loguniform('gamma', -16, 2),
            'lambdax':           hp.loguniform('lambdax', -16, 2),
            'alpha':             hp.loguniform('alpha', -16, 2)
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.CLASSIFICATION