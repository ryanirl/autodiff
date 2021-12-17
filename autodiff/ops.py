from autodiff.utils import broadcast
import numpy as np


@broadcast
class Add:
    def forward(x, y):
        return np.add(x.value, y.value)

    def backward(g, x, y, z):
        return [g, g]

@broadcast
class Sub:
    def forward(x, y):
        return x.value - y.value

    def backward(g, x, y, z):
        return [g, -g]

@broadcast
class Mul:
    def forward(x, y):
        return x.value * y.value

    def backward(g, x, y, z):
        g_x = g * y.value
        g_y = g * x.value

        return [g_x, g_y]

@broadcast
class Div:
    def forward(x, y):
        return x.value / y.value

    def backward(g, x, y, z):
        g_x = g * (1.0 / y.value)
        g_y = g * (-x.value / (y.value ** 2))

        return [g_x, g_y]

@broadcast
class Pow:
    def forward(x, y):
        return x.value ** y.value

    def backward(g, x, y, z):
        # Note: We do not compute the 'y' grad.
        g_x = g * y.value * np.power(x.value, y.value - 1)
        g_y = np.array([[0]])

        return [g_x, g_y]


### --- Basic Activation Function Ops --- ###

class Sigmoid:
    def forward(x):
        return 1.0 / (1.0 + np.exp(-x.value))

    def backward(g, x, z):
        return [g * (z.value * (1.0 - z.value))]

class ReLU:
    def forward(x):
        return np.maximum(x.value, 0)

    def backward(g, x, z):
        return [g * (x.value > 0)]

class Leaky_ReLU:
    def forward(x):
        return np.maximum(x.value, 0.1 * x.value)

    def backward(g, x, z):
        return [g * np.where(x.value > 0, 1, 0.1)]

class TanH:
    def forward(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def backward(g, x, z):
        return [g * (1.0 - (z.value ** 2))]


### --- Primitive Matrix Operations --- ###

class Sum:
    def forward(x):
        return np.sum(x.value, axis = x.axis, keepdims = x.keepdims)

    def backward(g, x, z):
        return [np.broadcast_to(g, x.shape_in)]

class Dot:
    def forward(x, y):
        return x.value.dot(y.value)

    def backward(g, x, y, z):
        return [g.dot(y.value.T), x.value.T.dot(g)]

class Transpose:
    def forward(x):
        return x.value.T

    def backward(g, x, z):
        return [g.T]

class Reshape:
    def forward(x):
        return np.reshape(x.value, x.new_shape)

    def backward(g, x, z):
        return [g.reshape(x.old_shape)]


### --- Elem-Wise Ops --- ###

class Log:
    def forward(x):
        return np.log(x.value)

    def backward(g, x, z):
        return [g / x.value]

class Exp:
    def forward(x):
        return np.exp(x.value)

    def backward(g, x, z):
        return [g * z.value]

class Abs:
    def forward(x):
        return np.abs(x.value)

    def backward(g, x, z):
        return [g * (z.value / np.where(x.value == 0, x.value, 1))]

class Max:
    def forward(x):
        return np.max(x.value)

    def backward(g, x, z):
        return [g * (x.value == z.value) / np.sum(x.value == z.value)]





