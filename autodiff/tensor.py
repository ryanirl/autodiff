# RYAN P 

import numpy as np
from autodiff.ops import grad_fun, value_fun
from autodiff.utils import check, _isscalar

class Tensor:
    def __init__(self, value, _children = [], requires_grad = True):
        self.value = value if isinstance(value, np.ndarray) else np.array(value)
        self.requires_grad = requires_grad
        self.grad = np.zeros(np.shape(self.value))

        self._outgrad = lambda x, y: () 
        self._children = _children

    # Operator Overloading

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        return OP("add", self, check(other, Tensor))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self)

    def __pow__(self, other):
        return OP("pow", self, check(other, Tensor))

    def __mul__(self, other):
        return OP("mul", self, check(other, Tensor))

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return OP("div", self, check(other, Tensor))

    def __rdiv__(self, other):
        return OP("div", check(other, Tensor), self)

    def __truediv__(self, other):
        return OP("div", self, check(other, Tensor))

    def __rtruediv__(self, other):
        return OP("div", check(other, Tensor), self)

    # Properties 

    @property
    def shape(self):
        return self.value.shape

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

    # Custom Ops

    def sum(self, axis = None, keepdims = True): 
        self.axis = axis
        self.shape_in = self.shape
        self.keepdims = keepdims
        return OP("sum", self)

    def transpose(self): # NOT TESTED
        return OP("transpose", self)

    def reshape(self, size):
        return OP("reshape", self, size)

    def sigmoid(self):
        return OP("sigmoid", self)

    def relu(self):
        return OP("relu", self)

    def log(self):
        return OP("log", self)
        
    def dot(self, other):
        return OP("dot", self, check(other, Tensor))

    def exp(self):
        return OP("exp", self)

    def softmax(self):
        return OP("softmax", self)

    def leaky_relu(self):
        return OP("leaky_relu", self)

    def tanh(self):
        return OP("tanh", self);

    # Backwards

    def backward(self):
        def recurse(tensor):
            grad = tensor._outgrad(tensor.grad, *tensor._children, tensor.value)

            for child, ingrad in zip(tensor._children, grad):
                child.grad = child.grad + ingrad

                recurse(child)

        self.grad = np.ones(np.shape(self.value))

        recurse(self)


### ----- OP TYEPS ----- ### 

def OP(op, *tensors):

    value = value_fun[op](*tensors)

    requires_grad = True if sum([tensor.requires_grad for tensor in tensors]) > 0 else False

    output_tensor = Tensor(value, tensors, requires_grad)

    output_tensor._outgrad = grad_fun[op]

    return output_tensor







