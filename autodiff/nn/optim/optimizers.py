from autodiff.nn.optim.base import Optimizer

import numpy as np


class SGD(Optimizer):
    def __init__(self, parameters, lr = 0.01):
        super().__init__(parameters, lr)

    def step(self):
        for param in self.parameters:
            param.value = param.value - (self.lr * param.grad)


class RMSProp(Optimizer):
    def __init__(self, parameters, lr = 0.01, eps = 1e-7, decay_rate = 0.99):
        super().__init__(parameters, lr)
        self.grad_squared = [np.zeros(tensor.shape) for tensor in parameters]
        self.decay_rate = 0.99
        self.eps = eps

    def step(self):
        for i, param in enumerate(self.parameters):
            self.grad_squared[i] = self.decay_rate * self.grad_squared[i] + (1.0 - self.decay_rate) * (param.grad * param.grad)

            param.value = param.value - self.lr * (param.grad / (np.sqrt(self.grad_squared[i]) + self.eps))


class AdaGrad(Optimizer):
    def __init__(self, parameters, lr = 0.01, eps = 1e-7):
        super().__init__(parameters, lr)
        self.grad_squared = [np.zeros(tensor.shape) for tensor in parameters]
        self.eps = eps

    def step(self):
        for i, param in enumerate(self.parameters):
            self.grad_squared[i] = self.grad_squared[i] + (param.grad * param.grad)

            ada = param.grad / (np.sqrt(self.grad_squared[i]) + self.eps)

            param.value = param.value - self.lr * ada


class Adam(Optimizer):
    def __init__(self, parameters, lr = 0.01, beta1 = 0.9, beta2 = 0.99, eps = 1e-7):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.moment1 = [np.zeros(tensor.shape) for tensor in parameters]
        self.moment2 = [np.zeros(tensor.shape) for tensor in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            self.moment1[i] = self.beta1 * self.moment1[i] + (1.0 - self.beta1) * param.grad
            self.moment2[i] = self.beta2 * self.moment2[i] + (1.0 - self.beta2) * (param.grad * param.grad)

            param.value = param.value - self.lr * (self.moment1[i] / (np.sqrt(self.moment2[i]) + self.eps))





