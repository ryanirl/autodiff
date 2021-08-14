# RYAN P 

import numpy as np
from autodiff.ops import grad_fun, value_fun
from autodiff.utils import check


class Tensor:
    def __init__(self, value, _children = [], requires_grad = True):
        self.value = np.atleast_2d(value) if isinstance(value, np.ndarray) else np.atleast_2d(value)
        self.requires_grad = requires_grad
        self.grad = np.zeros(np.shape(self.value))

        self._outgrad = lambda x, y: () 
        self._children = _children


    ### --- Operator Overloading --- ###

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


    ### --- Properties --- ###

    @property
    def shape(self):
        return self.value.shape

    @property
    def T(self):
        return OP("transpose", self)


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

    def sum(self, axis = None, keepdims = True): 
        self.axis = axis
        self.shape_in = self.shape
        self.keepdims = keepdims
        return OP("sum", self)

    def transpose(self): 
        return OP("transpose", self)

    def reshape(self, new_shape):
        self.old_shape = self.shape
        self.new_shape = new_shape
        return OP("reshape", self)

    def dot(self, other):
        return OP("dot", self, check(other, Tensor))

    
    ### --- Elem-Wise Ops --- ### 

    def log(self):
        return OP("log", self)

    def exp(self):
        return OP("exp", self)

    def abs(self):
        return OP("abs", self)

    def max(self):
        return OP("max", self)


    ### --- Activation Functions --- ###

    def sigmoid(self):
        return OP("sigmoid", self)

    def relu(self):
        return OP("relu", self)

    def softmax(self):
        return OP("softmax", self)

    def leaky_relu(self):
        return OP("leaky_relu", self)

    def tanh(self):
        return OP("tanh", self)

    # I optimized this because tensor_softmax 
    # is expensive so use softmax instead
    def tensor_softmax(self):
        a = (self - self.max()).exp()
        b = a.sum(axis = 1, keepdims = True)

        return a / b


    ### --- Loss Functions --- ###

    def stable_binary_cross_entropy_loss(self, actual):
        return OP("stable_binary_cross_entropy_loss", self, actual)

    def categorical_cross_entropy_loss(self, actual):
        return OP("categorical_cross_entropy_loss", self, actual)

    def sigmoid_binary_cross_entropy(self, actual):
        return OP("sigmoid_binary_cross_entropy", self, actual)

    def softmax_categorical_cross_entropy(self, actual):
        return OP("softmax_categorical_cross_entropy", self, actual)

    ### --- Backprop --- ###
    def backward(self):
        topo = []
        visited = set()
    
        def recurse(tensor):
            if tensor not in visited:
                visited.add(tensor)
    
                for child in tensor._children:
                   recurse(child)
    
#                topo.append(tensor)
                topo.insert(0, tensor)
    
        recurse(self)

        self.grad = np.ones(np.shape(self.value)) 

#        for tensor in reversed(topo):
        for tensor in topo:
            grad = tensor._outgrad(tensor.grad, *tensor._children, tensor.value)

            for child, ingrad in zip(tensor._children, grad):
                child.grad = child.grad + ingrad



### ----- OP TYEPS ----- ### 

def OP(op, *tensors):

    value = value_fun[op](*tensors)

    requires_grad = True if sum([tensor.requires_grad for tensor in tensors]) > 0 else False

    output_tensor = Tensor(value, tensors, requires_grad)

    output_tensor._outgrad = grad_fun[op]

    return output_tensor





