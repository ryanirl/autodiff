from autodiff.ops import grad_fun, value_fun
from autodiff.utils import check, broadcasted
from inspect import signature
import numpy as np


class Tensor:
    def __init__(self, value, _children = [], requires_grad = True):
        self.value = np.atleast_2d(value)
        self.requires_grad = requires_grad
        self.grad = np.zeros(np.shape(self.value))

        self._op = "LEAF"
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

    @broadcasted
    def __add__(self, other):
        return OP("add", self, check(other, Tensor))

    @broadcasted
    def __radd__(self, other):
        return OP("add", check(other, Tensor), self)

    @broadcasted
    def __sub__(self, other):
        return OP("sub", self, check(other, Tensor))

    @broadcasted
    def __rsub__(self, other): 
        return OP("sub", check(other, Tensor), self)

    @broadcasted
    def __pow__(self, other):
        return OP("pow", self, check(other, Tensor))

    @broadcasted
    def __mul__(self, other):
        return OP("mul", self, check(other, Tensor))

    @broadcasted
    def __rmul__(self, other):
        return OP("mul", check(other, Tensor), self)

    @broadcasted
    def __div__(self, other):
        return OP("div", self, check(other, Tensor))

    @broadcasted
    def __rdiv__(self, other):
        return OP("div", check(other, Tensor), self)

    @broadcasted
    def __truediv__(self, other):
        return OP("div", self, check(other, Tensor))

    @broadcasted
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

    def leaky_relu(self):
        return OP("leaky_relu", self)

    def tanh(self):
        return OP("tanh", self)


    ### --- Backprop & Computation Graph Functions --- ###

    def toposort(self):
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

    def topo_update(self):
        """
        If you update a leaf node, this may be used to update the rest
        of the graph along with it. That being forward then backwards
        propogating that change through the graph. If control flow allows
        for it, this would be an opimization to the naive 'backward()'
        method assuming large epochs.

        """
        for tensor in self._topo:
            tensor.value = tensor._forward(*tensor._children)
            tensor.grad = 0

        self.grad = np.ones(np.shape(self.value))

        for tensor in self._topo:
            grad = tensor._outgrad(tensor.grad, *tensor._children, tensor)

            for child, ingrad in zip(tensor._children, grad):
                child.grad = child.grad + ingrad

    def backward(self):
        self.toposort()

        self.grad = np.ones(np.shape(self.value))

        for tensor in self._topo:
            grad = tensor._outgrad(tensor.grad, *tensor._children, tensor)

            for child, ingrad in zip(tensor._children, grad):
                child.grad = child.grad + ingrad


### ----- OP BUILDER ----- ### 

def OP(op, *tensors):
    value = value_fun[op](*tensors)

    requires_grad = True if sum([tensor.requires_grad for tensor in tensors]) > 0 else False

    output_tensor = Tensor(value, tensors, requires_grad)

    output_tensor._outgrad = grad_fun[op]
    output_tensor._forward = value_fun[op]
    output_tensor._op = op

    return output_tensor


class primitive:
    def __new__(cls, *args, **kwargs):
        value_fun[cls.__name__] = (cls.forward)
        grad_fun[cls.__name__] = (cls.backward)

        parameters = list(signature(cls.forward).parameters)
        parameters[0] = 'self'

        method = lambda *parameters: OP(cls.__name__, *parameters)

        setattr(Tensor, cls.__name__, method) 

        return super().__new__(cls)





