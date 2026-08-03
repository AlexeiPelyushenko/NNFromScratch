import numpy as np
import yaml
import random, math
from easydict import EasyDict
from math_helper_funcs import *

with open("config.yaml", "r") as f:
    config = EasyDict(yaml.safe_load(f))


"""
Convention:

Everything should return numpy arrays if applicable.
Loss functions return floats.
"""

# --------------------------- Activation functions ---------------------------

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

def GELU(x):
    return x * standard_normal_cdf(x)

def GELU_d(x): # TODO: Make this use a cached forward pass result
    return standard_normal_cdf(x) + x * standard_normal_pdf(x)

def SiLU(x):
    return x * sigmoid(x)

def SiLU_d(x):
    sig_x = sigmoid(x)
    return sig_x * (1 + x * (1 - sig_x))

def softmax(x):
    ex = np.pow(np.e, x)
    return ex / sum(ex)

# --------------------------- Loss functions ---------------------------

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

# --------------------------- Distribution functions ---------------------------

def gaussian_boxmuller(n, mu=0, sigma=1, cutoff=None):
    """
    Uses the Box-Muller algorithm to generate a gaussian distribution.
    If cutoff is set to a value, points that are -cutoff- standard distributions away from the mean are removed.
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

def _init_helper(in_dim, out_dim, sigma, cutoff):
    std_correction = truncated_variance(cutoff)**0.5 if cutoff else 1
    sigma = sigma / std_correction
    if cutoff:
        results = gaussian_boxmuller(in_dim * out_dim, 0, sigma, cutoff)
    else:
        results = gaussian_boxmuller(in_dim * out_dim, 0, sigma)
    return results.reshape((in_dim, out_dim))

def he_initialization(in_dim, out_dim, cutoff=3): # good for linear unit activations
    """
    Kaiming He initialization. By default uses gaussian distribution with variance = 2/n_in and opts for truncating values
    > truncate standard deviations away from 0. Standard deviation is analytically scaled up to account for extreme values being removed.
    "If I throw away the tails of a Gaussian, how much variance do I lose?"
    
    cutoff: How many standard deviations away to not include points. Set to False/None to not truncate.
    """
    he_std = (2/in_dim)**0.5
    return _init_helper(in_dim, out_dim, he_std, cutoff)

def xavier_initialization(in_dim, out_dim, cutoff=3): # good for sigmoid/tanh
    """
    Xavier/Glorot initialization. Uses gaussian distribution with variance 2/(n_in + n_out)
    
    cutoff: How many standard deviations away to not include points. Set to False/None to not truncate.
    """
    xavier_std = (2/(in_dim + out_dim))**0.5
    return _init_helper(in_dim, out_dim, xavier_std, cutoff)

def lecun_initialization(in_dim, out_dim, cutoff=3): # good for SELU/linear
    """
    LeCun initialization. Uses gaussian distribution with variance 1/n_in)
    
    cutoff: How many standard deviations away to not include points. Set to False/None to not truncate.
    """
    lecun_std = (1/(in_dim))**0.5
    return _init_helper(in_dim, out_dim, lecun_std, cutoff)
    
def uniform_random_initialization(in_dim, out_dim, cutoff):
    return np.random.uniform(-1.0, 1.0, (in_dim, out_dim))

# --------------------------- Normalization functions ---------------------------

def min_max_normalization(data_train, data_test):
    data_min, data_max = data_train.min(axis=0), data_train.max(axis=0)
    scaling = data_max - data_min
    data_train = (data_train - data_min) / scaling
    data_test  = (data_test  - data_min) / scaling
    return data_train, data_test, scaling

def z_score_normalization(data_train, data_test): # predictions_original = predictions * y_std + y_mean
    data_mean, data_std = data_train.mean(axis=0), data_train.std(axis=0)
    data_train = (data_train - data_mean) / data_std
    data_test = (data_test - data_mean) / data_std
    return data_train, data_std, data_std


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    
    print(GELU(x))
    print(GELU_d(x))
    
    
    # num_true = 0
    # n = 10000
    # distribution = he_initialization(n, 10, cutoff = 3)
    # print(distribution)
    
    # variance = sum((distribution.mean() - distribution)**2) / n
    # print(variance)
    
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