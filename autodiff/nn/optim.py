from autodiff.tensor import Tensor
import numpy as np



class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0

    def add_parameters(self, new_param):
        self.parameters.append(new_param)



class SGD(Optimizer):
    def __init__(self, parameters, lr = 0.01, momentum = 0, nestrov = False):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.nestrov = nestrov

        self.velocity = np.array([np.zeros(tensor.shape) for tensor in parameters])

    def step(self):
        for i, param in enumerate(self.parameters):
            if not self.nestrov:
                if self.momentum != 0: 
                    self.velocity[i] = self.velocity[i] * self.momentum + param.grad

                param.value = param.value - self.lr * (param.grad + self.velocity[i])

            else: 
                old_velocity = self.velocity[i]

                self.velocity[i] = self.momentum * self.velocity[i] - (self.lr * param.grad)

                param.value = param.value + (-self.momentum * old_velocity + ((1.0 + self.momentum) * self.velocity[i]))



# NEED TO TEST
class RMSProp(Optimizer):
    def __init__(self, parameters, lr = 0.01, eps = 1e-7, decay_rate = 0.99):
        super().__init__(parameters, lr)
        self.eps = eps
        self.decay_rate = 0.99
        self.grad_squared = np.array([np.zeros(tensor.shape) for tensor in parameters])

    def step(self):
        for i, param in enumerate(self.parameters):
            grad = param.grad

            self.grad_squared[i] = self.decay_rate * self.grad_squared[i] + (1.0 - self.decay_rate) * (grad * grad)

            param.value = param.value - self.lr * (grad / (np.sqrt(self.grad_squared[i]) + self.eps))



# NEED TO TEST
class AdaGrad(Optimizer):
    def __init__(self, parameters, lr = 0.01, eps = 1e-7):
        super().__init__(parameters, lr)
        self.eps = eps
        self.grad_squared = np.array([np.zeros(tensor.shape) for tensor in parameters])

    def step(self):
        for i, param in enumerate(self.parameters):
            grad = param.grad

            self.grad_squared[i] = self.grad_squared[i] + (grad * grad)

            param.value = param.value - self.lr * (grad / (np.sqrt(self.grad_squared[i]) + self.eps))



# NEED TO TEST
class Adam(Optimizer):
    def __init__(self, parameters, lr = 0.01, beta1 = 0.9, beta2 = 0.99, eps = 1e-7):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.moment1 = np.array([np.zeros(tensor.shape) for tensor in parameters])
        self.moment2 = np.array([np.zeros(tensor.shape) for tensor in parameters])

    def step(self):
        for i, param in enumerate(self.parameters):
            grad = param.grad

            self.moment1[i] = self.beta1 * self.moment1[i] + (1.0 - self.beta1) * grad
            self.moment2[i] = self.beta2 * self.moment2[i] + (1.0 - self.beta2) * (grad * grad)

            param.value = param.value - self.lr * (self.moment1[i] / (np.sqrt(self.moment2[i]) + self.eps))








