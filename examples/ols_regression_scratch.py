from autodiff.tensor import Tensor

import numpy as np


# quite obvious the pattern here is y = 3x + 1
# Therefore out weight should = 3 and our bias should = 1
x = Tensor([1, 2, 3, 4, 5])
y = Tensor([4, 7, 10, 13, 16])


def ols_loss(y, pred):
    return (y - pred) ** 2


def pred(w, x, b):
    return w.dot(x) + b


class OLS:
    def __init__(self, dims):
        self.weights = Tensor.uniform(1, 1)
        self.bias = Tensor.uniform(1, 1)

    def zero_grad(self):
        self.weights.grad = 0
        self.bias.grad = 0
        self.x.grad = 0
        self.y.grad = 0

    def train(self, x, y, size, eta):
        self.x = x
        self.y = y

        for i in range(size):
            h = pred(self.weights, x, self.bias)

            loss = ols_loss(y, h)

            loss.backward()

            self.weights.value = self.weights.value - eta * self.weights.grad
            self.bias.value = self.bias.value - eta * self.bias.grad

            self.zero_grad()

        return self.weights.value, self.bias.value


model = OLS((1, 1))
w, b = model.train(x, y, 1000, 0.01)

print(f"{w}x + {b}")





