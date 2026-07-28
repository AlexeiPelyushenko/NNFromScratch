import numpy as np
from nn_funcs import *
import dill


class InputVec:
    def __init__(self, vals):
        self.vals = np.asarray(vals, dtype=float)

    def __matmul__(self, other):
        assert isinstance(other, Layer)
        return other.__rmatmul__(self)

    def __str__(self):
        return str(self.vals)
    
    def __sub__(self, other):
        return self.vals - other


class Layer:
    def __init__(self, dim, last_dim, forward_prop=sigmoid, backward_prop=sigmoid_d):
        self.weights = np.random.uniform(-1.0, 1.0, (last_dim, dim))
        self.biases = np.zeros(dim, dtype=float)

        self.forward_prop = forward_prop
        self.backward_prop = backward_prop

        self.last_input = None
        self.last_z = None
        self.last_output = None

    def __rmatmul__(self, other):
        assert isinstance(other, InputVec)
        return InputVec(self.forward(other.vals))

    def forward(self, input_vals):
        input_vals = np.asarray(input_vals, dtype=float)

        self.last_input = input_vals
        self.last_z = input_vals @ self.weights + self.biases
        self.last_output = self.forward_propagate(self.last_z)
        return self.last_output

    def backward(self, grad_output, learning_rate):
        grad_output = np.asarray(grad_output, dtype=float)

        if self.last_input is None or self.last_z is None:
            raise ValueError("forward() must be called before backward().")

        grad_z = grad_output * self.backward_prop(self.last_z)
        grad_weights = np.outer(self.last_input, grad_z)
        grad_biases = grad_z
        grad_input = grad_z @ self.weights.T

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

    def forward_propagate(self, input_vals):
        return self.forward_prop(input_vals)


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)

    layers = [
        Layer(2, 2),
        Layer(1, 2)
    ]

    learning_rate = 0.5
    epochs = 10000

    for epoch in range(epochs):
        total_loss = 0.0

        for x_vals, target in zip(X, y):
            out = InputVec(x_vals)
            for layer in layers:
                out = out @ layer

            error = out - target
            total_loss += np.mean(error ** 2)

            grad = 2 * error
            for layer in reversed(layers):
                grad = layer.backward(grad, learning_rate)

        if epoch % 1000 == 0:
            print(f"epoch {epoch:5d} | loss {total_loss / len(X):.6f}")

    print("\nFinal predictions:")
    for x_vals, target in zip(X, y):
        out = x_vals
        for layer in layers:
            out = layer.forward(out)

        print(f"{x_vals} -> {np.round(out, 3)} target={target}")