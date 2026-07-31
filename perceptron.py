import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n, learning_rate=0.01):
        self.weights = np.array([0.0]*n)
        self.bias = 0.0
        self.learning_rate = learning_rate
        
    def infer(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
    
    def train(self, inputs, correct_output):
        output = self.infer(inputs)
        inputs = np.array(inputs)
        if np.sign(output) != np.sign(correct_output):
            self.weights = self.weights + self.learning_rate * correct_output * inputs
            self.bias = self.bias + self.learning_rate * correct_output
            
            
if __name__ == "__main__":
    points = [(1, 1), (2, 1), (1.5, 1.2), (2, 0.5), (3, 1), (4, 5), (5, 4), (4.5, 5.5), (6, 5), (5, 6)]
    correct_classes = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
    
    p = Perceptron(2)
    
    for i in range(100):
        for j in range(len(points)):
            p.train(points[j], correct_classes[j])
            
    print(p.weights)
    print(p.bias)
    
    for j in range(len(points)):
        print(f"Test for point {points[j]}")
        print(p.infer(points[j]))
        print(correct_classes[j])

    points = np.array(points)
    correct_classes = np.array(correct_classes)

    neg = points[correct_classes == -1]
    pos = points[correct_classes == 1]

    plt.scatter(neg[:, 0], neg[:, 1], c="blue", label="class -1")
    plt.scatter(pos[:, 0], pos[:, 1], c="red", label="class +1")

    # Decision boundary: w1*x + w2*y + b = 0  =>  y = -(w1*x + b) / w2
    w1, w2 = p.weights
    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    xs = np.linspace(x_min, x_max, 200)
    ys = -(w1 * xs + p.bias) / w2
    plt.plot(xs, ys, "k-", label="decision boundary")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Perceptron Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
