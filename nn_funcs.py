import numpy as np
import yaml
from easydict import EasyDict

with open("config.yaml", "r") as f:
    config = EasyDict(yaml.safe_load(f))


"""
Convention:

Activation functions + their derivatives should return numpy arrays.
Loss functions return floats.
"""

identity = lambda x: x
identity_d = lambda x: np.ones_like(x)

def sigmoid(x):
    return 1 / (1 + np.e**(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

def RELU(x):
    return np.maximum(0, x)

def RELU_d(x):
    return (x > 0).astype(float)

def leaky_RELU(x):
    return np.where(x > 0, x, config.LEAKY_RELU_COEFF * x)

def leaky_RELU_d(x):
    return np.where(x > 0, 1.0, config.LEAKY_RELU_COEFF)

def softmax(x):
    ex = np.pow(np.e, x)
    return ex / sum(ex)

def MSE(x, y):
    return np.mean((x - y)**2)

def MSE_grad(x, y):
    return 2 * (x - y)

def cross_entropy(p, y, softmax_preprocess=True):
    """
    Gradient: p - y (if combining softmax)
    
    Calculate the cross entropy loss of a prediction
    p: predictions
    y: true label
    """
    if softmax_preprocess:
        p = softmax(p)
    return -np.sum(y * np.log(p))


if __name__ == "__main__":
    y = np.array([0, 1, 0])
    p = np.array([0.2, 0.7, 0.1])
    print(MSE(p, y))