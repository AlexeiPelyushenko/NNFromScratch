import numpy as np
from layers import *
from nn_funcs import *

class Network:
    """
    Front facing interface for the user.
    """
    
    def __init__(self, loss_func, loss_func_grad, epochs, learning_rate, input_vector_dim):
        self.loss_func = loss_func
        self.loss_func_grad = loss_func_grad
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.layers = []
        self.input_vector_dim = input_vector_dim
        
    
    def add_layer(self, layer_type, dim, forward, backward):
        if not self.layers:
            last_dim = self.input_vector_dim
        else:
            last_dim = self.layers[-1].dim
        
        if layer_type == "FF":
            self.layers.append(FFLayer(dim, last_dim, forward, backward, self.learning_rate))
            
    
    def inference(self, input_vec):
        out = InputVec(input_vec)
        for layer in self.layers:
            out = out @ layer
        return out.vals
    
    
    def backpropagate(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    
    def train(self, X, y, num_prints=10, num_sample_results=10):
        assert len(self.layers) > 0
        assert len(X) > 0
        assert len(X) == len(y)
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            for x_vals, target in zip(X, y):
                prediction = self.inference(x_vals)
                total_loss += self.loss_func(prediction, target)
                
                grad = self.loss_func_grad(prediction, target)
                self.backpropagate(grad)
                
            if epoch % (self.epochs // num_prints) == 0:
                print(f"epoch {epoch:5d} | loss {total_loss / len(X):.6f}")
                
        print("\nFinal predictions:")
        counter = 0
        while counter < num_sample_results:
            out = self.inference(X[counter])
            print(*np.round(X[counter], 3), sep=", ", end=", ")
            print(f"-> prediction={np.round(out, 3)} target={round(y[counter], 3)}")
            counter += 1
                
                
                
if __name__ == "__main__":
    nn = Network(MSE, MSE_grad, 10000, 0.02, 8)
    
    nn.add_layer("FF", 64, sigmoid, sigmoid_d)
    nn.add_layer("FF", 16, sigmoid, sigmoid_d)
    nn.add_layer("FF", 4, sigmoid, sigmoid_d)
    nn.add_layer("FF", 1, identity, identity_d)
    
    result = nn.inference([8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23])
    print(result)