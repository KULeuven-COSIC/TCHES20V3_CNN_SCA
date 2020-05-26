import numpy as np
from sklearn import preprocessing


def no_preprocessing(X_profiling, X_attack):
    X_profiling_processed = np.copy(X_profiling)
    X_attack_processed = np.copy(X_attack)
    
    return X_profiling_processed, X_attack_processed


# Feature scaling between 0 and 1 as was used by Zaid et al.
# This method is not recomended.
def feature_scaling_0_1(X_profiling, X_attack):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    X_profiling_processed = scaler.fit_transform(X_profiling)
    X_attack_processed = scaler.transform(X_attack)

    return X_profiling_processed, X_attack_processed


def feature_scaling_m1_1(X_profiling, X_attack):
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

    X_profiling_processed = scaler.fit_transform(X_profiling)
    X_attack_processed = scaler.transform(X_attack)

    return X_profiling_processed, X_attack_processed



def feature_standardization(X_profiling, X_attack):
    scaler = preprocessing.StandardScaler()

    X_profiling_processed = scaler.fit_transform(X_profiling)
    X_attack_processed = scaler.transform(X_attack)
    
    return X_profiling_processed, X_attack_processed


def horizontal_scaling_0_1(X_profiling, X_attack):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_profiling.T)
    X_profiling_processed = scaler.transform(X_profiling.T).T

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_attack.T)
    X_attack_processed = scaler.transform(X_attack.T).T
    
    return X_profiling_processed, X_attack_processed


def horizontal_scaling_m1_1(X_profiling, X_attack):
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X_profiling.T)
    X_profiling_processed = scaler.transform(X_profiling.T).T

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X_attack.T)
    X_attack_processed = scaler.transform(X_attack.T).T
    
    return X_profiling_processed, X_attack_processed


def horizontal_standardization(X_profiling, X_attack):
    mn = np.repeat(np.mean(X_profiling, axis=1, keepdims=True), X_profiling.shape[1], axis=1)
    std = np.repeat(np.std(X_profiling, axis=1, keepdims=True), X_profiling.shape[1], axis=1)
    X_profiling_processed = (X_profiling - mn)/std

    mn = np.repeat(np.mean(X_attack, axis=1, keepdims=True), X_attack.shape[1], axis=1)
    std = np.repeat(np.std(X_attack, axis=1, keepdims=True), X_attack.shape[1], axis=1)
    X_attack_processed = (X_attack - mn)/std
    
    return X_profiling_processed, X_attack_processed
