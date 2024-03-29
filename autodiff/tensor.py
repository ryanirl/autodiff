from autodiff.utils import check
from autodiff.ops import *

from inspect import signature
import numpy as np


class Tensor:
    def __init__(self, value, _children = [], requires_grad = True):
        self.value = np.atleast_2d(value)
        self.requires_grad = requires_grad
        self.grad = np.zeros(np.shape(self.value))

        self._op = "leaf"
        self._forward = lambda: self.value
        self._outgrad = lambda x, y: () 
        self._children = _children


    ### --- Tensor Representation --- ### 

    def __str__(self):
        return "< {} Tensor | Value : {} | Grad: {} | Instant at: {} >".format(
                  self._op, self.value, self.grad, id(self))

    __repr__ = __str__


    ### --- Operator Overloading --- ###

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        return OP(Add, self, check(other, Tensor))

    def __radd__(self, other):
        return OP(Add, check(other, Tensor), self)

    def __sub__(self, other):
        return OP(Sub, self, check(other, Tensor))

    def __rsub__(self, other): 
        return OP(Sub, check(other, Tensor), self)

    def __pow__(self, other):
        return OP(Pow, self, check(other, Tensor))

    def __mul__(self, other):
        return OP(Mul, self, check(other, Tensor))

    def __rmul__(self, other):
        return OP(Mul, check(other, Tensor), self)

    def __div__(self, other):
        return OP(Div, self, check(other, Tensor))

    def __rdiv__(self, other):
        return OP(Div, check(other, Tensor), self)

    def __truediv__(self, other):
        return OP(Div, self, check(other, Tensor))

    def __rtruediv__(self, other):
        return OP(Div, check(other, Tensor), self)


    ### --- Properties --- ###

    @property
    def shape(self):
        return self.value.shape

    @property
    def T(self):
        return OP(Transpose, self)


    ### --- Class Methods --- ###

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(shape), **kwargs)

    @classmethod
    def uniform(cls, *shape, **kwargs):
        return cls((np.random.uniform(-1, 1, size = shape)), **kwargs)

    @classmethod
    def eye(cls, dims, **kwargs):
        return cls(np.eye(dims), **kwargs)

    @classmethod
    def diag(cls, x, k = 0, **kwargs):
        return cls(np.diag(x, k), **kwargs)


    ### --- Matrix Ops --- ###

    def sum(self, **kwargs):
        return OP(Sum, self, **kwargs)

    def transpose(self): 
        return OP(Transpose, self)

    def reshape(self, *shape, **kwargs):
        return OP(Reshape, self, *shape, **kwargs)

    def dot(self, other):
        return OP(Dot, self, check(other, Tensor))

    
    ### --- Elem-Wise Ops --- ### 

    def log(self):
        return OP(Log, self)

    def exp(self):
        return OP(Exp, self)

    def abs(self):
        return OP(Abs, self)

    def max(self):
        return OP(Max, self)


    ### --- Activation Functions --- ###

    def sigmoid(self):
        return OP(Sigmoid, self)

    def relu(self):
        return OP(ReLU, self)

    def leaky_relu(self):
        return OP(Leaky_Relu, self)

    def tanh(self):
        return OP(TanH, self)


    ### --- Backprop & Computation Graph Functions --- ###

    def topo_sort(self):
        topo = []
        visited = set()

        def recurse(tensor):
            if tensor not in visited:
                visited.add(tensor)
    
                for child in tensor._children:
                   recurse(child)
    
                topo.insert(0, tensor)
    
        recurse(self)

        self._topo = topo

    def backward(self):
        self.topo_sort()

        self.grad = np.ones(np.shape(self.value))

        for tensor in self._topo:
            # This is a GOTCHA of the current implementation, even if
            # requires_grad = True, the gradient still gets computed. 
            grad = tensor._outgrad(tensor.grad, *tensor._children, tensor)

            for child, ingrad in zip(tensor._children, grad):
                if child.requires_grad:
                    child.grad = child.grad + ingrad


### ----- OP BUILDER ----- ### 

def OP(op, *args, **kwargs):
    value = op.forward(*args, **kwargs)

    tensors = [arg for arg in args if isinstance(arg, Tensor)]

    requires_grad = True if np.any([tensor.requires_grad for tensor in tensors]) else False

    output_tensor = Tensor(value, tensors, requires_grad)

    output_tensor._outgrad = op.backward
    output_tensor._forward = op.forward
    output_tensor._op = op.__name__.lower()

    return output_tensor


class Function:
    def __new__(cls, *args, **kwargs):
        parameters = list(signature(cls.forward).parameters)
        parameters[0] = "self"

        method = lambda *parameters: OP(cls, *parameters)

        setattr(Tensor, cls.__name__, method) 

        return super().__new__(cls)


def register(cls):
    cls = type(cls.__name__, (cls, Function), {})

    cls()

    return cls





