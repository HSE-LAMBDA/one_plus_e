from sklearn.base import BaseEstimator
# from sklearn.metrics import *
from evaluation import get_anomaly_metrics
from keras.utils import to_categorical
import sys
from hmc import *
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer
from layers import WeightNormLayer
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy
import lasagne
import numpy as np

class OPE(BaseEstimator):
    def __init__(self, pseudo_neg_size, alpha=None, hidden_layer_size=50, anomaly_label=1, lr=5e-4, 
                 epoches=10, batch_size=10000, sampler_epoch_rate=1, verbose=0):
        self._hidden_layer_size = hidden_layer_size
        self._pseudo_neg_size = pseudo_neg_size
        self._anomaly_label = anomaly_label
        self._lr = lr
        self._alpha = alpha
        self._epoches = epoches
        self._batch_size = batch_size
        self._sampler_epoch_rate = sampler_epoch_rate
        self._verbose = verbose
    
    def fit(self, X, y):
        # Neural network
        input_size = X.shape[1] # features cnt
        
        X_min, X_max = np.min(X, axis=0), np.max(X, axis=0) # sampling range
        
        def random_sampling(size, features_range=[X_min, X_max]):
            return np.array([features_range[0][i] * np.ones(size) + np.random.rand(size) * (features_range[1][i] - features_range[0][i]) for i in range(len(features_range[0]))]).transpose()

        input_var = T.matrix('input', dtype='float32')
        weights_var = T.vector('weights', dtype='float32')
        target_var = T.matrix('target', dtype='float32')
        lr_var = T.scalar('learning rate')

        self._network = InputLayer(shape=(None, input_size), input_var=input_var)
        self._network = DenseLayer(self._network, self._hidden_layer_size)
        self._network = WeightNormLayer(self._network)
        self._network = DenseLayer(self._network, 2, nonlinearity=softmax)
        self._output = lasagne.layers.get_output(self._network)

        loss = (weights_var * categorical_crossentropy(self._output, target_var)).mean()
        self._params = lasagne.layers.get_all_params(self._network, trainable=True)
        updates = lasagne.updates.rmsprop(loss, self._params, learning_rate=lr_var)

        train_fn = theano.function([input_var, weights_var, target_var, lr_var], [self._output, loss], updates=updates, allow_input_downcast=True)
        self._predict_fn = theano.function([input_var], [self._output], allow_input_downcast=True)
        
        
        X_good = X[y != self._anomaly_label]
        X_bad_true = X[y == self._anomaly_label]
        
        if self._anomaly_label != 1: y = 1 - y
        
        
        good_size = X_good.shape[0]
        true_bad_size = X_bad_true.shape[0]
        fake_bad_size = self._pseudo_neg_size
#         alpha = true_bad_size / (true_bad_size + fake_bad_size) # balancing heuristics
        alpha = self._alpha
        if alpha is None: alpha = true_bad_size * 1.0 / fake_bad_size
        
        X_bad_fake = random_sampling(fake_bad_size)


        target = np.concatenate([np.zeros(good_size), np.ones(true_bad_size + fake_bad_size)])
        batch_size = min(target.shape[0], self._batch_size)
        good_weights = np.ones(X_good.shape[0])
        
        true_bad_weights = alpha * np.ones(true_bad_size)
        fake_bad_weights = (1 - alpha) * np.ones(fake_bad_size)

        position = theano.shared(X_bad_fake)

        def P(x):
            preds = x
            for layer in lasagne.layers.get_all_layers(self._network)[1:]: # pass x variable through all network layers except input one
                preds = layer.get_output_for(preds)
            preds = preds[:, 1]
            return 1 - T.exp(-preds / (1 - preds))

        def get_P(x):
            preds = self._predict_fn(x)[0].squeeze()[:, 1]
            return 1 - np.exp(-preds / (1 - preds))

        self._sampler = HMC_sampler.new_from_shared_positions(position, P,
                              initial_stepsize=1e-3, stepsize_max=0.5)

        if self._verbose > 0:
            X_test = np.concatenate([X_good, X_bad_true])
            y_test = np.concatenate([np.zeros(good_size), np.ones(true_bad_size)])

        
        for epoch in range(self._epoches):
            # TODO: lr decay
            [self._sampler.draw() for _ in range(self._sampler_epoch_rate)]

            X_bad_fake = np.concatenate([self._sampler.draw() for _ in range(int(round(fake_bad_size / batch_size)))])[:fake_bad_size]
            X_mix = np.concatenate([X_good, X_bad_fake, X_bad_true])

            eps = 1e-4
            fake_bad_weights = 1./(eps + get_P(X_bad_fake))
#             fake_bad_weights /= max(fake_bad_weights) # normalization for nans
            fake_bad_weights *= (1. - alpha)
            

            weights = np.concatenate([good_weights, fake_bad_weights, true_bad_weights])
            indices = np.arange(len(weights))
            np.random.shuffle(indices)
            for i in range(int(len(indices)/batch_size)):
                batch_idx = indices[range(batch_size*i,min(batch_size*(i+1), len(indices)))]
                _, loss_value = train_fn(X_mix[batch_idx], weights[batch_idx], to_categorical(target[batch_idx], num_classes=2), self._lr)
            
            # Debug
            if self._verbose > 0:
                y_pred = self.predict_proba(X_test).squeeze()[:, 1]
                metrics = get_anomaly_metrics(y_test, y_pred)
                print(metrics)
        
        
    def predict_proba(self, X):
        return self._predict_fn(X)[0]
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
