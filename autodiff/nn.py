import numpy as np
from autodiff.tensor import Tensor, OP
from autodiff.ops import grad_fun, value_fun
from autodiff.utils import primitive, check
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs

# NOT TESTED

class Linear:
    def __init__(self, dims_in, dims_out, bias = True):
        self.weight = Tensor.uniform(dims_in, dims_out)
        self.needs_bias = bias

        if self.needs_bias == True: self.bias = Tensor.zeros(dims_out)

    def __call__(self, X):
        if self.needs_bias: return X.dot(self.weight) + self.bias
        else: return X.dot(self.weight)

class ReLU:
    def __call__(self, X): return X.relu()

class Sigmoid:
    def __call__(self, X): return X.sigmoid()

class Sequential:
    def __init__(self, *layers): 
        self.layers = layers

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)

        return X


if __name__ == "__main__":
    X, y = make_blobs(n_samples = 100, centers = 2)

    X = Tensor(X)
    y = Tensor(np.array(y)[:, np.newaxis])

    model = Sequential(
        Linear(2, 1),
        Sigmoid()
    )


    pred = model(X)

    print(pred.value)
















