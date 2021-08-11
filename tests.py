import numpy as np
from autodiff.tensor import Tensor, OP
from autodiff.ops import grad_fun, value_fun
from autodiff.utils import primitive, check
import autodiff.nn as nn
import time 

# This is a mess 
def softmaxCCE():
    a = Tensor(np.array([[1, 2, 3], [1, 2, 3]]))
    b = Tensor(np.array([[0, 1, 0], [1, 0, 0]]))

    loss_func = nn.CrossEntropy() 

    c = a.softmax()

    loss = loss_func(c, b)

    loss.backward()

    print(c.grad)
    print(a.grad)



def test0():
    # Test
    # Should be:
    # Value: 22500
    # df/dA: 310500
    # df/dB: 242996
    # df/dC: -3

    a = Tensor(2)
    b = Tensor(3)
    c = Tensor(4)

    z = a + b
    y = z ** 2
    x = y * b
    w = x * a
    v = w ** 2
    u = a * b
    t = v * u
    s = c * b
    r = t - s

    r.backward()

    print("value: ({}) should be 134988".format(r.value))
    print("grad wrt A: ({}) should be 310500".format(a.grad))
    print("grad wrt B: ({}) should be 242996".format(b.grad))
    print("grad wrt C: ({}) should be -3".format(c.grad))

def speedtest():

    # This is a standard test for me:
    # Average run time using topo sort and old backwards method: 7.4s over 5 runs
    # Average run time using new method without topo sort: 6.3s over 5 runs
    # Literally cut off a second which is quite large in the grand scheme of things

    # This takes 3 seconds to run 100,000 iterations without autodiff and implimenting 
    # it by hand with computing the gradient of OLS. Meaning that it only takes twice
    # the time using AutoDiff and not being told it's gradient. That's quite impressive

    start = time.time()

    def OLS_loss(w, x, b, y):
        return (y - (w * x + b)) ** 2

    class Neuron:
        def __init__(self, loss, dims):
            self.weights = Tensor([np.random.uniform() for i in range(dims)])
#            self.weights = Tensor.uniform(1, 1)
            self.bias = Tensor([np.random.uniform()])
#            self.bias = Tensor.uniform(1, 1)
            self.dims = dims
            self.loss = loss

        def forward(self, x, y):
            self.y = y
            self.x = x

        def step(self, itter, eta):
            for i in range(itter):
                self.loss(self.weights, self.x, self.bias, self.y).backward()

                self.weights.value = self.weights.value - eta * self.weights.grad.sum()
                self.bias.value = self.bias.value - eta * self.bias.grad.sum()

                self.weights.grad = 0
                self.bias.grad = 0
                self.x.grad = 0
                self.y.grad = 0

            return self.weights, self.bias

    x = Tensor([0, 1, 3], requires_grad = False)
    y = Tensor([1, 4, 10], requires_grad = False)

    neuron = Neuron(OLS_loss, 1)
    neuron.forward(x, y)
    weight, bias = neuron.step(100000, 0.01)
    print(weight.value)
    print(bias.value)

    end = time.time()

    print(f"Runtime of the program is {end - start}")

def test3():
    def e(x):
        """
        You don't need this I am just using it to simplify
        the value function.

        """
        return np.exp(x.value)

    value_fun["tanh"] = (lambda x: (e(x) - e(-x)) / (e(x) + e(-x)))

    # multiplying each gradient by "g" is requied by the chain rule
    grad_fun["tanh"] = (lambda g, x, z: [(g * (1.0 -(z ** 2)))])

    # Check(x, Tenor) just garentees that some x is not Tensor
    @primitive(Tensor)
    def tanh(self):
        return OP("tanh", self);

    x = Tensor([1, 2, 3])
    y = x.tanh()

    y.backward()

    print(f"The gradient of x: {x.grad}")



if __name__ == "__main__":
    softmaxCCE()
    test0()
    speedtest()
    test3()









