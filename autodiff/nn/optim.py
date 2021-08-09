from autodiff.tensor import Tensor
import numpy as np

# SGD
# MBSGD

# MOMENTUM
# ADAM
# RMSPROP
# ADAGRAD
# Nestrov MOMENTUM
# BFGS


class SGD:
    def __init__(self, parameters, lr = 0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0

    def step(self):
        for param in self.parameters:
            param.value = param.value - self.lr * param.grad










