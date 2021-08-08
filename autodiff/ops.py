import numpy as np
from collections import defaultdict

grad_fun = defaultdict(lambda: "ERROR")
value_fun = defaultdict(lambda: "ERROR")

######### --- TRIVIAL OPS --- #########
##### --- GRAD DEFINITIONS --- #####

# Binary
grad_fun["add"] = (lambda g, x, y, z: (g, g))
grad_fun["sub"] = (lambda g, x, y, z: (g, -g))
grad_fun["mul"] = (lambda g, x, y, z: (g * y.value, g * x.value))
grad_fun["pow"] = (lambda g, x, y, z: (g * np.power(y.value * x.value, y.value - 1), 0))
grad_fun["div"] = (lambda g, x, y, z: (g / y.value, g / (y.value ** 2)))


# Unary
# Haven't extensivly tested these yet so be cautious.
grad_fun["sigmoid"] = (lambda g, x, z: [g * (z * (1.0 - z))])
grad_fun["relu"] = (lambda g, x, z: [g * (x.value > 0)]) 
grad_fun["log"] = (lambda g, x, z: [g / x.value])
grad_fun["exp"] = (lambda g, x, z: [(g * z)])



##### --- VALUE DEFINITIONS --- #####

# Binary
value_fun["add"] = (lambda x, y: x.value + y.value)
value_fun["sub"] = (lambda x, y: x.value - y.value)
value_fun["mul"] = (lambda x, y: x.value * y.value)
value_fun["div"] = (lambda x, y: x.value / y.value)
value_fun["pow"] = (lambda x, y: x.value ** y.value)

# Unary
value_fun["relu"] = (lambda x: np.maximum(x.value, 0)) # NEED TO TEST
value_fun["sigmoid"] = (lambda x: 1.0 / (1.0 + np.exp(-x.value))) # NEED TO TEST
value_fun["log"] = (lambda x: np.log(x.value))
value_fun["exp"] = (lambda x: (np.exp(x.value)))

### ---  END PRIM  --- ###


### --- STILL NEED TO DO SOME TESTING --- ###
grad_fun["dot"] = (lambda g, x, y, z: (np.dot(g, y.value.T), np.dot(x.value.T, g)))
value_fun["dot"] = (lambda x, y: np.dot(x.value, y.value))

#value_fun["sum"] = (lambda x: (x.value.sum(axis = 1, keepdims = True)))
#grad_fun["sum"] = (lambda g, x, z: (g * x.cache, ))



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






grad_fun["leaky_relu"] = (lambda g, x, z: [g * np.where(x.value > 0, 1, 0.1)])

value_fun["leaky_relu"] = (lambda x: np.maximum(x.value, 0.1 * x.value)) # NEED TO TEST





def e(x): return np.exp(x.value)

value_fun["tanh"] = (lambda x: (e(x) - e(-x)) / (e(x) + e(-x)))

grad_fun["tanh"] = (lambda g, x, z: [(g * (1.0 - (z ** 2)))])





