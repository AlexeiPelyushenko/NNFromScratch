import numpy as np
import yaml
import random, math
from easydict import EasyDict
from math_helper_funcs import *

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

def gaussian_boxmuller(n, mu=0, sigma=1, cutoff=None):
    """
    Uses the Box-Muller algorithm to generate a gaussian distribution.
    If truncate is set to a value, points that are -truncate- standard distributions away from the mean are removed.
    """
    result = []
    while len(result) < n:
        u1, u2 = random.random(), random.random()
        r = math.sqrt(-2 * math.log(u1))
        theta = 2 * math.pi * u2
        
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        
        if cutoff:
            if abs(z0) <= cutoff:
                result.append(z0)
            if abs(z1) <= cutoff:
                result.append(z1)
        else:
            result.extend([z0, z1])
        
    if len(result) > n:
        result.pop()
    
    return mu + sigma * np.array(result)

def he_initialization(in_dim, out_dim, cutoff=3):
    """
    Kaiming He initialization. By default uses gaussian distribution with variance = 2/n_in and opts for truncating values
    > truncate standard deviations away from 0. Standard deviation is analytically scaled up to account for extreme values being removed.
    "If I throw away the tails of a Gaussian, how much variance do I lose?"
    
    cutoff: How many standard deviations away to not include points. Set to False/None to not truncate.
    """
    
    std_correction = truncated_variance(cutoff)**0.5 if cutoff else 1
    he_std = (2/in_dim)**0.5
    sigma = he_std / std_correction
    if cutoff:
        results = gaussian_boxmuller(in_dim * out_dim, 0, sigma, cutoff)
    else:
        results = gaussian_boxmuller(in_dim * out_dim, 0, sigma)
    return results.reshape((in_dim, out_dim))

def uniform_random_initialization(in_dim, out_dim):
    return np.random.uniform(-1.0, 1.0, (in_dim, out_dim))


if __name__ == "__main__":
    num_true = 0
    n = 10000
    distribution = he_initialization(n, 10, cutoff = 3)
    print(distribution)
    
    variance = sum((distribution.mean() - distribution)**2) / n
    print(variance)
    
    # print("mean:", distribution.mean())
    # print("he-variance", 2/n)
    # print("variance:", variance)
    # print("he-std", (2/n)**0.5)
    # print("std:", variance**0.5)
    
    
    # if test_variance:
    #     n = 10000
    #     num_true = 0
    #     cutoff = 3
    #     # std_correction = truncated_variance(cutoff)**0.5 if cutoff else 1
    #     std_correction = 0.9865783925581086
    #     for i in range(20):
    #         distribution = gaussian_boxmuller(n, 0, 1, cutoff)
    #         variance = sum((distribution - distribution.mean())**2) / len(distribution)
            
    #         if variance > 1:
    #             num_true += 1
        
    #     print("Variance was greater than 1:", num_true)
    #     print("Variance was less than 1:", 20 - num_true)