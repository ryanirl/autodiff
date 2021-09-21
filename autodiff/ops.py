import numpy as np
from collections import defaultdict

from autodiff.utils import _unbroadcast
from autodiff.utils import to_logits, clip_stable
from autodiff.utils import _conv2d_forward, _conv2d_backward

grad_fun = defaultdict(lambda: "ERROR")
value_fun = defaultdict(lambda: "ERROR")

# I need to remove all of the overhead of forward and just have everything
# needed set at the first creation of the first value. This can either be done
# with an if statement in "OP" for broadcasted ops, or ...??? 


### --- Primitive Operator Overloaded Ops --- ###

#value_fun["LEAF"] = (lambda x: x.value)

value_fun["add"] = (lambda x, y: x.value + y.value)
grad_fun["add"] = (lambda g, x, y, z: [_unbroadcast(g, z._unbroadcast_axis[0], x.shape), 
                                       _unbroadcast(g, z._unbroadcast_axis[1], y.shape)])
#grad_fun["add"] = (lambda g, x, y, z: (g, g))

value_fun["sub"] = (lambda x, y: x.value - y.value)
grad_fun["sub"] = (lambda g, x, y, z: [g, -g])

value_fun["mul"] = (lambda x, y: x.value * y.value)
grad_fun["mul"] = (lambda g, x, y, z: [_unbroadcast(g * y.value, z._unbroadcast_axis[0], x.shape), 
                                       _unbroadcast(g * x.value, z._unbroadcast_axis[1], y.shape)])
#grad_fun["mul"] = (lambda g, x, y, z: (g * y.value, g * x.value))

# NOTE: Does not take the derivative W.R.T. y
value_fun["pow"] = (lambda x, y: x.value ** y.value)
grad_fun["pow"] = (lambda g, x, y, z: [_unbroadcast(g * y.value * np.power(x.value, y.value - 1), z._unbroadcast_axis[0], x.shape), 0])
#grad_fun["pow"] = (lambda g, x, y, z: (g * np.power(y.value * x.value, y.value - 1), 0))

# I re-wrote div in terms of exponentiation ?TEMP?
value_fun["div"] = (lambda x, y: x.value / y.value)
grad_fun["div"] = (lambda g, x, y, z: (_unbroadcast(g * (1.0 / y.value), z._unbroadcast_axis[0], x.shape), 
                                       _unbroadcast(g * (-x.value / (y.value ** 2)), z._unbroadcast_axis[1], y.shape)))


### --- Activation Function Ops --- ###

grad_fun["sigmoid"] = (lambda g, x, z: [g * (z.value * (1.0 - z.value))])
value_fun["sigmoid"] = (lambda x: 1.0 / (1.0 + np.exp(-x.value))) 

value_fun["relu"] = (lambda x: np.maximum(x.value, 0)) 
grad_fun["relu"] = (lambda g, x, z: [g * (x.value > 0)]) 

grad_fun["leaky_relu"] = (lambda g, x, z: [g * np.where(x.value > 0, 1, 0.1)])
value_fun["leaky_relu"] = (lambda x: np.maximum(x.value, 0.1 * x.value)) # NEED TO TEST

value_fun["tanh"] = (lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))
grad_fun["tanh"] = (lambda g, x, z: [(g * (1.0 - (z.value ** 2)))])


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

grad_fun["exp"] = (lambda g, x, z: [g * z.value])
value_fun["exp"] = (lambda x: (np.exp(x.value)))

value_fun["abs"] = (lambda x: np.abs(x.value))
grad_fun["abs"] = (lambda g, x, z: [g * (z.value / np.where(x.value == 0, x.value, 1))])


# NEED TO TEST THIS I HAVE NO IDEA IF IT IS VALID 
grad_fun["max"] = (lambda g, x, z: [g * (x.value == z.value) / np.sum(x.value == z.value)])
value_fun["max"] = (lambda x: np.max(x.value))

# NEED TO DO
#grad_fun["amax"]
#value_fun["amax"]


### --- NON-TRIVIAL OPS --- ###

def optimize_softmax(x): a = np.exp(x.value - np.max(x.value)); return a / np.sum(a, axis = 1, keepdims = True)
value_fun["softmax"] = (lambda x: optimize_softmax(x))
grad_fun["softmax"] = (lambda g, x, z: [(g * z.value) - (np.einsum('ijk,ik->ij', z.value[..., None] * z.value[:, None, :], g))])
#value_fun["softmax"] = (lambda x: np.exp(x.value - np.max(x.value)) / np.sum(np.exp(x.value - np.max(x.value)), axis = 1, keepdims = True))
#grad_fun["softmax"] = (lambda g, x, z: [softmax_backward_helper(g, z)])


def optimize_bce(pred, actual): pred = to_logits(pred); return np.sum(np.maximum(pred, 0) - (pred * actual) + np.log(1.0 + np.exp(-np.abs(pred))))
value_fun["stable_binary_cross_entropy_loss"] = (lambda x, y: optimize_bce(x.value, y.value))
grad_fun["stable_binary_cross_entropy_loss"] = (lambda g, pred, actual, z: (g * ((1.0 / (1.0 + np.exp(-to_logits(pred.value)))) - actual.value), ))


value_fun["categorical_cross_entropy_loss"] = (lambda pred, actual: -np.sum(actual.value * np.log(clip_stable(pred.value)), axis = 1, keepdims = True))
grad_fun["categorical_cross_entropy_loss"] = (lambda g, pred, actual, z: ((-actual.value / (pred.value + 1e-6)), ))


value_fun["sigmoid_binary_cross_entropy"] = (lambda pred, actual: (1.0 / pred.shape[0]) * np.sum(
                                             np.maximum(pred.value, 0) - (pred.value * actual.value) + np.log(1.0 + np.exp(-np.abs(pred.value)))))

grad_fun["sigmoid_binary_cross_entropy"] = (lambda g, pred, actual, z: (g * ((1.0 / (1.0 + np.exp(-pred.value))) - actual.value), ))


# These assume one-hot ecoded data
value_fun["softmax_categorical_cross_entropy"] = (lambda pred, actual: -np.sum(actual.value * np.log(clip_stable(softmax_forward(pred.value))), axis = 1, keepdims = True))
grad_fun["softmax_categorical_cross_entropy"] = (lambda g, pred, actual, z: [g * (softmax_forward(pred.value) - actual.value), ])


### --- Convolution Ops --- ### 

value_fun["conv2d"] = (lambda x, weights: _conv2d_forward(x, weights.value))
grad_fun["conv2d"] = (lambda g, x, weights, z: _conv2d_backward(g, x, weights, z.value))


#value_fun["pool2d"] = (lambda x: x)
#grad_fun["pool2d"] = (lambda g, x, z: g)





