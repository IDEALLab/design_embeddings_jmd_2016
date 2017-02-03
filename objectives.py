from keras import backend as K
import numpy as np

def smape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true) + K.abs(y_pred), K.epsilon(), np.inf))
    return K.mean(diff)
    