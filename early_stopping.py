"""
Early stopping for the autoencoder.

Author(s): Wei Chen (wchen459@umd.edu)
"""

from keras.callbacks import Callback, EarlyStopping
import warnings
import numpy as np

class MyEarlyStopping(EarlyStopping):

    def __init__(self, monitor='loss', patience=0, verbose=0, mode='min', tol=1e-4):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.tol = tol
        self.wait = 0
        self.first = 0
        self.second = 0
        self.previous = np.inf
        
    def on_train_begin(self, logs={}):
        self.wait = 0       # Allow instances to be re-used
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % (self.monitor), RuntimeWarning)
            
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if abs(self.previous-current) > self.tol*self.previous:
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.stop_training = True
            self.wait += 1

        self.previous = current
