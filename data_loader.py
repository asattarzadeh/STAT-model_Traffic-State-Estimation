import numpy as np
import pandas as pd

def load_data(data_path, seq_len=15, his=1, pre_sens_num=1):
    data = pd.read_hdf(data_path)
    max_val = np.max(data, axis=0)
    min_val = np.min(data, axis=0)
    med = max_val - min_val
    data = np.array(data, dtype=float)
    data_nor = (data - min_val) / med

    sequence_length = seq_len + his
    result = []
    for index in range(len(data_nor) - sequence_length):
        result.append(data_nor[index: index + sequence_length])
    result = np.stack(result, axis=0)
    x_train = result[:, :seq_len]
    x_wd_train = result[:, :seq_len, pre_sens_num-1]
    y_train = result[:, -1, :]
    return x_train, x_wd_train, y_train, med*.95, min_val

def generate_data(data_path, seq_len, pre_len, pre_sens_num):
    x_train, x_wd_train, y_train, med, min_val = load_data(data_path, seq_len, pre_len, pre_sens_num)
    row = 2016
    train_x_data = x_train[:-row]
    test_data = x_train[-row:]
    train_w = x_wd_train[:-row]
    test_w = x_wd_train[-row:]
    train_l = y_train[:-row]
    test_l = y_train[-row:]
    return train_x_data, train_w, train_l, test_data, test_w, test_l, med, min_val
