import numpy as np
from autodiff.tensor import Tensor, OP
from autodiff.ops import grad_fun, value_fun
from autodiff.utils import primitive, check
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs

# NOT YET TESTED

params = []
class Linear:
    def __init__(self, dims_in, dims_out, bias = True):
        self.weight = Tensor.uniform(dims_in, dims_out)
        params.append(self.weight)
        self.needs_bias = bias

        if self.needs_bias == True: 
            self.bias = Tensor.zeros(dims_out)
            params.append(self.bias)

    def __call__(self, X):
        if self.needs_bias: return X.dot(self.weight) + self.bias
        else: return X.dot(self.weight)

class Sequential:
    def __init__(self, *layers): 
        self.layers = layers

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)

        return X

    def zero_grad(self):
        for p in params:
            p.grad = 0


    def step(self):
        for p in params:
            p.value = p.value - 0.01 * p.grad

        self.zero_grad()

if __name__ == "__main__":
    X, y = make_blobs(n_samples = 100, centers = 2)

    X = Tensor(X)
    y = Tensor(np.array(y)[:, np.newaxis])

    ## SPLIT ## 

    Y = y.value
    X = X.value
    print(np.shape(X))

    for i in range(100):
        if (Y[i] == 1):
            plt.scatter(X[i][0], X[i][1], color="green") 
        else:
            plt.scatter(X[i][0], X[i][1], color="blue") 

        
    x = np.linspace(-20, 20, 20)
    hyperplane = ((-(weight0 / weight1) * x) - (bias/weight1))

    plt.plot(x, hyperplane, '-', color="blue")
    plt.show()

















