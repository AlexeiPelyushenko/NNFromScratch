import numpy as np
import sys, os

parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(parent))

from layer import Layer, InputVec
from nn_funcs import *

if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    # one-hot labels for 2-class XOR
    y = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ], dtype=float)

    identity = lambda x: x
    identity_d = lambda x: 1.0

    layers = [
        Layer(2, 2),  # hidden: sigmoid
        # output: linear logits (softmax applied outside for CE)
        Layer(2, 2)
    ]

    epochs = 10000

    for epoch in range(epochs):
        total_loss = 0.0

        for x_vals, target in zip(X, y):
            out = InputVec(x_vals)
            for layer in layers:
                out = out @ layer

            logits = out.vals
            probs = softmax(logits)
            total_loss += cross_entropy(logits, target)

            # fused softmax + CE gradient w.r.t. logits: p - y
            grad = probs - target
            for layer in reversed(layers):
                grad = layer.backward(grad)

        if epoch % 1000 == 0:
            print(f"epoch {epoch:5d} | loss {total_loss / len(X):.6f}")

    print("\nFinal predictions:")
    for x_vals, target in zip(X, y):
        out = x_vals
        for layer in layers:
            out = layer.forward(out)

        probs = softmax(out)
        pred = np.argmax(probs)
        true = np.argmax(target)
        print(f"{x_vals} -> probs={np.round(probs, 3)} pred={pred} target={true}")
