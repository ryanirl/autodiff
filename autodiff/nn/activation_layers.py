from autodiff.tensor import Tensor
from autodiff.nn.containers import Module

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



