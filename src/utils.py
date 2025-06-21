import pickle
import numpy as np
import os

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    w = np.array(model['weights'])
    b = model['bias']
    return w, b

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    mean = np.array(scaler['mean'])
    std = np.array(scaler['std'])
    return mean, std

def scale_features(X, mean, std):
    return (X - mean) / std 