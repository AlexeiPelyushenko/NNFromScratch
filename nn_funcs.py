import numpy as np


def sigmoid(x):
    return 1 / (1 + np.e**(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

def RELU(x):
    return max(0, x)

def RELU_d(x):
    return 1 if x > 0 else 0

def leaky_RELU(x):
    return 1 if x > 0 else 0.01*x

def softmax(v):
    ex = np.pow(np.e, v)
    return ex / sum(ex)


if __name__ == "__main__":
    v = np.array([1, 2, 3, 4])
    print(np.pow(np.e, v))
    print(softmax(v))