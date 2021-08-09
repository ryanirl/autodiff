import numpy as np
from autodiff.tensor import Tensor, OP
from autodiff.ops import grad_fun, value_fun
from autodiff.utils import primitive, check


class Module:
    def __init__(self):
        self.params = []

    def parameters(self):
        return self.params


class Linear(Module):
    def __init__(self, dims_in, dims_out, bias = True):
        super().__init__()

        self.weight = Tensor.uniform(dims_in, dims_out)
        self.params.append(self.weight)
        self.needs_bias = bias

        if self.needs_bias == True: 
            self.bias = Tensor.uniform(dims_out)
            self.params.append(self.bias)

    def __call__(self, X):
        if self.needs_bias: return X.dot(self.weight) + self.bias
        else: return X.dot(self.weight)

class Sequential(Module):
    def __init__(self, *layers): 
        self.params = []
        self.layers = layers

        for layer in layers:
            self.params += layer.parameters()

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)

        return X


####### --------- ACTIVATION LAYERS --------- ####### 

class ReLU(Module):
    def __call__(self, X): return X.relu()

class Sigmoid(Module):
    def __call__(self, X): return X.sigmoid()

class Softmax(Module):
    def __call__(self, X): return X.softmax()

class LeakyReLU(Module):
    def __call__(self, X): return X.leaky_relu()

class Tanh(Module):
    def __call__(self, X): return X.tanh()

class Sigmoid(Module):
    def __call__(self, X): return X.sigmoid()

##################################################### 







