from sklearn.datasets import fetch_california_housing
import os, sys

parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(parent))

from network import Network, train_test_split
from nn_funcs import *


if __name__ == "__main__":
    # df = fetch_california_housing(as_frame=True).frame
    X, y = fetch_california_housing(return_X_y=True)
    
    # Normalize data
    # min_vals = raw_X.min(axis=0)
    # max_vals = raw_X.max(axis=0)
    # X = (raw_X - min_vals) / (max_vals - min_vals)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, seed=None, shuffle=True)
    X_train, X_test, _ = min_max_normalization(X_train, X_test)
    y_train, y_test, scaling = min_max_normalization(y_train, y_test)

    nn = Network(MSE, MSE_grad, 25, 0.001, 8)
    nn.add_layer("FF", 256, leaky_RELU, leaky_RELU_d, he_initialization)
    nn.add_layer("FF", 128, leaky_RELU, leaky_RELU_d, he_initialization)
    nn.add_layer("FF", 1, identity, identity_d)
    
    
    nn.train(X_train, y_train, scaling, num_sample_results=30)
    print("----")
    nn.test(X_test, y_test, scaling)