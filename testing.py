#from inspect import signature
#from autodiff.tensor import Tensor, primitive
#
#class softy(primitive):
#    def forward(x, y):
#        return x.value + y.value
#
#    def backward(g, x, y, z):
#        return g
#
#
#
#
#
#
#a = Tensor([1, 2, 3])
#b = Tensor([1, 2, 3])
#
#softy()
#
#
#print(a)
#a = a.softy(b)
#a.backward()
#print(a.grad)
#print(a)
#print(signature(Tensor.softy))
#
#
#
#
#
#
#


from autodiff.tensor import Tensor, primitive
import numpy as np

# For simplicity
def e(x): return np.exp(x.value)

class tanh(primitive):
    def forward(x):
        return (e(x) - e(-x)) / (e(x) + e(-x))

    def backward(g, x, z):
        return [g * (1.0 - (z.value ** 2))]


# Must do this so it 'registers'
tanh()

x = Tensor([1, 2, 3])
y = x.tanh()

y.backward()

print("The gradient of y wrt x: {}".format(x.grad))

