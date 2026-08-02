from sklearn.datasets import fetch_california_housing
import os, sys

parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(parent))

from network import Network
from nn_funcs import *


if __name__ == "__main__":
    # df = fetch_california_housing(as_frame=True).frame
    raw_X, y = fetch_california_housing(return_X_y=True)
    
    # Normalize data
    min_vals = raw_X.min(axis=0)
    max_vals = raw_X.max(axis=0)
    X = (raw_X - min_vals) / (max_vals - min_vals)

    nn = Network(MSE, MSE_grad, 25, 0.02, 8)
    nn.add_layer("FF", 256, leaky_RELU, leaky_RELU_d)
    nn.add_layer("FF", 128, sigmoid, sigmoid_d)
    nn.add_layer("FF", 1, identity, identity_d)
    
    nn.train(X, y, num_sample_results=30)