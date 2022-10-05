import numpy as np


def check(x, Type): 
    return x if isinstance(x, Type) else Type(x)


### --- Unbroadcasting Functions --- ### 

def unbroadcast_axes(shape_in, shape_out):
    """
    Main Idea: Given an ingrad with different shape from the gradient of our
    current value we must re-shape the ingrad to match the shape of our current
    gradient.

    ---

    Given: shape_in: (3, 2) & shape_out: (3, 1)
    
    Sum along axis (1) bec 2 -> 1 at index 1.

    ---

    Given shape_in: (3, 4, 1, 5, 3) & shape_out: (1, 4, 1, 1, 3)

    Sum along axis (0, 3) because index's 0, 2, and 3 shrunk to 1 and were not
    originally 1.

    """
    if shape_out == (): return None

    reduction_axes = []
    for i in range(len(shape_in)):
        if (shape_in[i] > shape_out[i]) & (shape_out[i] == 1):
            reduction_axes += [i]

    return tuple(reduction_axes)


def _unbroadcast(grad, axis, shape): 
    return np.reshape(np.sum(grad, axis = axis, keepdims = True), shape)


def broadcast(cls):
    class Function:
        def forward(x, y):
            x.broadcasted, y.broadcasted = False, False

            out = cls.forward(x, y)

            if x.shape != out.shape:
                x.broadcasted = True
                x.unbroadcast_axes = unbroadcast_axes(out.shape, x.shape)

            if y.shape != out.shape:
                y.broadcasted = True
                y.unbroadcast_axes = unbroadcast_axes(out.shape, y.shape)

            return out

        def backward(g, x, y, z):
            g_x, g_y = cls.backward(g, x, y, z)

            if x.broadcasted:
                g_x = _unbroadcast(g_x, x.unbroadcast_axes, x.shape)

            if y.broadcasted:
                g_y = _unbroadcast(g_y, y.unbroadcast_axes, y.shape)

            return [g_x, g_y]
    
    Function.__name__ = cls.__name__

    return Function 





