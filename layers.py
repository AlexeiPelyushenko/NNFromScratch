import numpy as np
from nn_funcs import *
from abc import ABC, abstractmethod


class BaseLayer(ABC):
    def __init__(self, dim, last_dim, learning_rate=0.2):
        self.dim = dim
        self.last_dim = last_dim
        self.learning_rate = learning_rate
        
        # Variables for backprop
        self.last_input = None
        self.last_z = None
        self.last_output = None
        
        
    @abstractmethod
    def __rmatmul__(self, other):
        """
        A "matmul" between an InputVec @ Layer represents passing in the input vector into the layer and processing one forward step
        in the neural network.
        
        other: InputVec
        return: Output of layer as InputVec
        """
        pass
    
    
    @abstractmethod
    def forward(self, input_vals):
        """
        Defines what the forward pass of this layer performs.
        
        input_vals: InputVec, output of previous layer or initial input.
        """
        pass
    
    
    @abstractmethod
    def backward(self, grad_output):
        """
        Defines the backward pass of this layer; takes a gradient output of a layer and uses it to update its own weights.
        """
        pass


class InputVec:
    """
    Defined this class for the purpose of using dunder methods to make NN implementation easier/more intuitive. Functionally is about
    the same as a numpy.array besides the __matmul__ override.
    """
    def __init__(self, vals):
        self.vals = np.asarray(vals, dtype=float)

    def __matmul__(self, other):
        assert isinstance(other, BaseLayer)
        return other.__rmatmul__(self)

    def __str__(self):
        return str(self.vals)
    
    def __sub__(self, other):
        return self.vals - other
    

class FFLayer(BaseLayer):
    """
    Feedforward layer of a neural network.
    """
    def __init__(self, dim, last_dim, forward_prop=sigmoid, backward_prop=sigmoid_d, learning_rate=0.5):
        super().__init__(dim, last_dim, learning_rate)
        self.weights = np.random.uniform(-1.0, 1.0, (last_dim, dim))
        self.biases = np.zeros(dim, dtype=float)

        self.forward_prop = forward_prop
        self.backward_prop = backward_prop
        

    def __rmatmul__(self, other):
        assert isinstance(other, InputVec)
        return InputVec(self.forward(other.vals))


    def forward(self, input_vals):
        input_vals = np.asarray(input_vals, dtype=float)

        self.last_input = input_vals
        self.last_z = input_vals @ self.weights + self.biases
        self.last_output = self.forward_prop(self.last_z)
        return self.last_output


    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=float)

        if self.last_input is None or self.last_z is None:
            raise ValueError("forward() must be called before backward().")

        grad_z = grad_output * self.backward_prop(self.last_z)
        grad_weights = np.outer(self.last_input, grad_z)
        grad_biases = grad_z
        grad_input = grad_z @ self.weights.T

        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases

        return grad_input