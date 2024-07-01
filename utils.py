import numpy as np
from tensorflow.keras import backend as K

def rmse_train(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def my_loss(y_true, y_pred):
    L1 = K.sum(K.abs(y_true - y_pred))
    L2 = K.sum(K.square(y_true - y_pred))
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    return L1 + L2 + mse

def predict_point_by_point(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    print("predict_size:", predicted.size)
    return predicted

def MAE(pre, true):
    return np.mean(np.abs(pre - true))

def MAPE(pre, true):
    return np.mean(np.abs((pre - true) / true))

def RMSE(pre, true):
    return np.sqrt(np.mean((pre - true) ** 2))
