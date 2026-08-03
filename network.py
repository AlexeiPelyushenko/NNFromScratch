import numpy as np
from layers import *
from nn_funcs import *

class Network:
    """
    Front facing interface for the user.
    """
    
    def __init__(self, loss_func, loss_func_grad, epochs, learning_rate, input_vector_dim, std_cutoff=3):
        self.loss_func = loss_func
        self.loss_func_grad = loss_func_grad
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.std_cutoff = std_cutoff
        
        self.layers = []
        self.input_vector_dim = input_vector_dim
        
    
    def add_layer(self, layer_type, dim, forward, backward, initialization=uniform_random_initialization):
        if not self.layers:
            last_dim = self.input_vector_dim
        else:
            last_dim = self.layers[-1].dim
        
        if layer_type == "FF":
            self.layers.append(FFLayer(dim, last_dim, forward, backward, initialization, self.learning_rate, self.std_cutoff))
            
    
    def inference(self, input_vec):
        out = InputVec(input_vec)
        for layer in self.layers:
            out = out @ layer
        return out.vals
    
    
    def backpropagate(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    
    def train(self, X, y, scaling=1, num_prints=10, num_sample_results=10, shuffle=True):
        assert len(self.layers) > 0
        assert len(X) > 0
        assert len(X) == len(y)
        
        for epoch in range(self.epochs):
            total_loss = 0
            X_epoch, y_epoch = shuffle_data(X, y) if shuffle else (X, y)
            
            for x_vals, target in zip(X_epoch, y_epoch):
                prediction = self.inference(x_vals)
                total_loss += self.loss_func(prediction, target)
                
                grad = self.loss_func_grad(prediction, target)
                self.backpropagate(grad)
                
            if epoch % (self.epochs // num_prints) == 0:
                print(f"epoch {epoch:5d} | Training loss {total_loss / len(X) * scaling ** 2:.6f}")
                
        print("\nFinal predictions:")
        counter = 0
        while counter < num_sample_results:
            out = self.inference(X[counter])
            print(*np.round(X[counter], 3), sep=", ", end=", ")
            print(f"-> prediction={np.round(out, 3)} target={round(y[counter], 3)}")
            counter += 1
            
    
    def test(self, X, y, scaling=1):
        total_loss = 0
        for x_vals, target in zip(X, y):
            prediction = self.inference(x_vals)
            total_loss += self.loss_func(prediction, target)
        print("Average test loss:", total_loss/len(X) * scaling**2)
                

def shuffle_data(X, y, seed=None):
    assert type(X) == np.ndarray
    assert type(y) == np.ndarray
    assert len(X) == len(y)
    
    indices = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return X[indices], y[indices]


def train_test_split(X, y, test_size=0.2, seed=None, shuffle=True):
    assert type(X) == np.ndarray
    assert type(y) == np.ndarray
    assert len(X) == len(y)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        
    n_test = int(np.ceil(n_samples * test_size))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return (X[train_indices], X[test_indices], y[train_indices], y[test_indices])
    
                
if __name__ == "__main__":
    nn = Network(MSE, MSE_grad, 10000, 0.02, 8)
    
    nn.add_layer("FF", 64, sigmoid, sigmoid_d)
    nn.add_layer("FF", 16, sigmoid, sigmoid_d)
    nn.add_layer("FF", 4, sigmoid, sigmoid_d)
    nn.add_layer("FF", 1, identity, identity_d)
    
    result = nn.inference([8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23])
    print(result)