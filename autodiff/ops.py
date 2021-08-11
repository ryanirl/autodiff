import numpy as np
from collections import defaultdict
from autodiff.utils import _unbroadcast

grad_fun = defaultdict(lambda: "ERROR")
value_fun = defaultdict(lambda: "ERROR")


### --- Primitive Operator Overloaded Ops --- ###

#grad_fun["add"] = (lambda g, x, y, z: (g, g))
value_fun["add"] = (lambda x, y: x.value + y.value)
grad_fun["add"] = (lambda g, x, y, z: [_unbroadcast(g, x.shape), _unbroadcast(g, y.shape)])

grad_fun["sub"] = (lambda g, x, y, z: [g, -g])
value_fun["sub"] = (lambda x, y: x.value - y.value)

#grad_fun["mul"] = (lambda g, x, y, z: (g * y.value, g * x.value))
grad_fun["mul"] = (lambda g, x, y, z: [_unbroadcast(g * y.value, x.shape), _unbroadcast(g * x.value, y.shape)])
value_fun["mul"] = (lambda x, y: x.value * y.value)

# NOTE: Does not take the derivative W.R.T. y
#grad_fun["pow"] = (lambda g, x, y, z: (g * np.power(y.value * x.value, y.value - 1), 0))
grad_fun["pow"] = (lambda g, x, y, z: [_unbroadcast(g * y.value * np.power(x.value, y.value - 1), x.shape), 0])
value_fun["pow"] = (lambda x, y: x.value ** y.value)

# I re-wrote div in terms of exponentiation TEMP
value_fun["div"] = (lambda x, y: x.value / y.value)
grad_fun["div"] = (lambda g, x, y, z: (_unbroadcast(g * (1.0 / y.value), x.shape), _unbroadcast(g * (-x.value / (y.value ** 2)), y.shape)))



### --- Activation Function Ops --- ###

grad_fun["sigmoid"] = (lambda g, x, z: [g * (z * (1.0 - z))])
value_fun["sigmoid"] = (lambda x: 1.0 / (1.0 + np.exp(-x.value))) 

value_fun["relu"] = (lambda x: np.maximum(x.value, 0)) 
grad_fun["relu"] = (lambda g, x, z: [g * (x.value > 0)]) 

grad_fun["leaky_relu"] = (lambda g, x, z: [g * np.where(x.value > 0, 1, 0.1)])
value_fun["leaky_relu"] = (lambda x: np.maximum(x.value, 0.1 * x.value)) # NEED TO TEST

value_fun["tanh"] = (lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))
grad_fun["tanh"] = (lambda g, x, z: [(g * (1.0 - (z ** 2)))])



### --- Matrix Operations --- ###

# NEED TO TEST
value_fun["sum"] = (lambda x: np.sum(x.value, axis = x.axis, keepdims = x.keepdims))
grad_fun["sum"] = (lambda g, x, z: [np.broadcast_to(g, x.shape_in)])

grad_fun["dot"] = (lambda g, x, y, z: (np.dot(g, y.value.T), np.dot(x.value.T, g)))
value_fun["dot"] = (lambda x, y: np.dot(x.value, y.value))

# This works the same I'm pretty sure. Need to test if one is faster than the other
#grad_fun["dot"] = (lambda g, x, y, z: (g @ y.value.T, x.value.T @ g))
#value_fun["dot"] = (lambda x, y: x.value @ y.value)

value_fun["transpose"] = (lambda x: x.value.T)
grad_fun["transpose"] = (lambda g, x, z: [g.T])

#grad_fun["reshape"] = (lambda g, x, z: [np.reshape(g, x.old_shape)])
value_fun["reshape"] = (lambda x: np.reshape(x.value, x.new_shape))
grad_fun["reshape"] = (lambda g, x, z: g.reshape(x.old_shape))



### --- Elem-Wise Ops --- ###

grad_fun["log"] = (lambda g, x, z: [g / x.value])
value_fun["log"] = (lambda x: np.log(x.value))

grad_fun["exp"] = (lambda g, x, z: [g * z])
value_fun["exp"] = (lambda x: (np.exp(x.value)))











# I NEED TO UPDATE THIS

######### --- NON-TRIVIAL OPS --- #########

def softmax_forward(data):
    a = np.exp(data - np.max(data))
    b = np.sum(np.exp(data - np.max(data)), axis = 1, keepdims = True)

    forward_out = a / b

    return forward_out

def softmax_backward(ingrad, forward_out):
    """
    I did NOT write this code.

    This code was found here: https://themaverickmeerkat.com/2019-10-23-Softmax/

    BIG thanks, these einsum ops genius.
    
    """

    m, n = forward_out.shape

    p1 = np.einsum('ij,ik->ijk', forward_out, forward_out)

    p2 = np.einsum('ij,jk->ijk', forward_out, np.eye(n, n))  

    jacobian = p2 - p1 

    out = np.einsum('ijk,ik->ij', jacobian, ingrad)  

    return out 

value_fun["softmax"] = (lambda x: softmax_forward(x.value))

grad_fun["softmax"] = (lambda g, x, z: [softmax_backward(g, z)])






















