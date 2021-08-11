# RYAN P 

import numpy as np
from autodiff.ops import grad_fun, value_fun
from autodiff.utils import check, _isscalar


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


    ### --- Class Methods --- ###

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


    ### --- Matrix Ops --- ###

    def sum(self, axis = None, keepdims = True): 
        self.axis = axis
        self.shape_in = self.shape
        self.keepdims = keepdims
        return OP("sum", self)

    def transpose(self): # NOT TESTED
        return OP("transpose", self)

    def reshape(self, new_shape):
        self.old_shape = self.shape
        self.new_shape = new_shape
        return OP("reshape", self)

    def dot(self, other):
        return OP("dot", self, check(other, Tensor))

    
    # I could probably add axis param with these. 
    ### --- Elem-Wise Ops --- ### 

    def log(self):
        return OP("log", self)

    def exp(self):
        return OP("exp", self)


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
        return OP("tanh", self);


    ### --- Backprop --- ###

    def backward(self):
        topo = []
        visited = set()
    
        def recurse(tensor):
            if tensor not in visited:
                visited.add(tensor)
    
                for child in tensor._children:
                   recurse(child)
    
                topo.append(tensor)
    
        recurse(self)
    
        self.grad = np.ones(np.shape(self.value)) 

        for tensor in reversed(topo):
            grad = tensor._outgrad(tensor.grad, *tensor._children, tensor.value)

            for child, ingrad in zip(tensor._children, grad):
                child.grad = child.grad + ingrad


#    def backward(self):
#        """
#        It seems this doesn't quite work in one case :(  I wonder if there is
#        something else like this in which is more computationally efficient
#        than doing a topo sort before hand
#        """
#        def recurse(tensor):
#            grad = tensor._outgrad(tensor.grad, *tensor._children, tensor.value)
#
#            for child, ingrad in zip(tensor._children, grad):
#                child.grad = child.grad + ingrad
#
#                recurse(child)
#
#        self.grad = np.ones(np.shape(self.value))
#
#        recurse(self)


### ----- OP TYEPS ----- ### 

def OP(op, *tensors):

    value = value_fun[op](*tensors)

    requires_grad = True if sum([tensor.requires_grad for tensor in tensors]) > 0 else False

    output_tensor = Tensor(value, tensors, requires_grad)

    output_tensor._outgrad = grad_fun[op]

    return output_tensor





