from collections import defaultdict
import numpy as np


value_fun = defaultdict(lambda: "ERROR")
grad_fun = defaultdict(lambda: "ERROR")


### --- Unbroadcasting Function --- ### 

# unbroadcasting is extremely slow. Removing it increases speed by 30%
def _unbroadcast(grad, axis, shape): return np.reshape(np.sum(grad, axis = axis, keepdims = True), shape)

# For reference using this lambda function instead is a direct
# grad_fun["add"] = (lambda g, x, y, z: [g, g])
# 30% performance increase on `bench.py`


### --- Primitive Mathematical Operators --- ###

value_fun["add"] = (lambda x, y: x.value + y.value)
grad_fun["add"] = (lambda g, x, y, z: [_unbroadcast(g, z._unbroadcast_axis[0], x.shape), 
                                       _unbroadcast(g, z._unbroadcast_axis[1], y.shape)])

value_fun["sub"] = (lambda x, y: x.value - y.value)
grad_fun["sub"] = (lambda g, x, y, z: [_unbroadcast(g, z._unbroadcast_axis[0], x.shape), 
                                       _unbroadcast(-g, z._unbroadcast_axis[1], y.shape)])

value_fun["mul"] = (lambda x, y: x.value * y.value)
grad_fun["mul"] = (lambda g, x, y, z: [_unbroadcast(g * y.value, z._unbroadcast_axis[0], x.shape), 
                                       _unbroadcast(g * x.value, z._unbroadcast_axis[1], y.shape)])

# NOTE: Does not take the derivative W.R.T. 'y'
value_fun["pow"] = (lambda x, y: x.value ** y.value)
grad_fun["pow"] = (lambda g, x, y, z: [_unbroadcast(g * y.value * np.power(x.value, y.value - 1), z._unbroadcast_axis[0], x.shape), 
                                       0])

value_fun["div"] = (lambda x, y: x.value / y.value)
grad_fun["div"] = (lambda g, x, y, z: [_unbroadcast(g * (1.0 / y.value), z._unbroadcast_axis[0], x.shape), 
                                       _unbroadcast(g * (-x.value / (y.value ** 2)), z._unbroadcast_axis[1], y.shape)])


### --- Basic Activation Function Ops --- ###

value_fun["sigmoid"] = (lambda x: 1.0 / (1.0 + np.exp(-x.value))) 
grad_fun["sigmoid"] = (lambda g, x, z: [g * (z.value * (1.0 - z.value))])

value_fun["relu"] = (lambda x: np.maximum(x.value, 0)) 
grad_fun["relu"] = (lambda g, x, z: [g * (x.value > 0)]) 

value_fun["leaky_relu"] = (lambda x: np.maximum(x.value, 0.1 * x.value)) 
grad_fun["leaky_relu"] = (lambda g, x, z: [g * np.where(x.value > 0, 1, 0.1)])

value_fun["tanh"] = (lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))
grad_fun["tanh"] = (lambda g, x, z: [g * (1.0 - (z.value ** 2))])


### --- Primitive Matrix Operations --- ###

value_fun["sum"] = (lambda x: np.sum(x.value, axis = x.axis, keepdims = x.keepdims))
grad_fun["sum"] = (lambda g, x, z: [np.broadcast_to(g, x.shape_in)])

#value_fun["dot"] = (lambda x, y: np.dot(x.value, y.value))
value_fun["dot"] = (lambda x, y: x.value.dot(y.value))
grad_fun["dot"] = (lambda g, x, y, z: [g.dot(y.value.T), x.value.T.dot(g)])
#grad_fun["dot"] = (lambda g, x, y, z: [np.dot(g, y.value.T), np.dot(x.value.T, g)])

value_fun["transpose"] = (lambda x: x.value.T)
grad_fun["transpose"] = (lambda g, x, z: [g.T])

value_fun["reshape"] = (lambda x: np.reshape(x.value, x.new_shape))
grad_fun["reshape"] = (lambda g, x, z: [g.reshape(x.old_shape)])


### --- Elem-Wise Ops --- ###

value_fun["log"] = (lambda x: np.log(x.value))
grad_fun["log"] = (lambda g, x, z: [g / x.value])

value_fun["exp"] = (lambda x: (np.exp(x.value)))
grad_fun["exp"] = (lambda g, x, z: [g * z.value])

value_fun["abs"] = (lambda x: np.abs(x.value))
grad_fun["abs"] = (lambda g, x, z: [g * (z.value / np.where(x.value == 0, x.value, 1))])

value_fun["max"] = (lambda x: np.max(x.value))
grad_fun["max"] = (lambda g, x, z: [g * (x.value == z.value) / np.sum(x.value == z.value)])





