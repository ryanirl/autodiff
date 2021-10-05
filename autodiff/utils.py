import numpy as np


### --- Tensor Utils --- ###

def check(x, Type): 
    return x if isinstance(x, Type) else Type(x)


def primitive(Class):
    def register_methods(method):
        setattr(Class, method.__name__, method) 
        return method 

    return register_methods 


def broadcasted(func): 
    def _op(x, y):
        output_tensor = func(x, y)

        x = output_tensor._children[0]
        y = output_tensor._children[1]

        x_axis = unbroadcast_axes(output_tensor.shape, x.shape)
        y_axis = unbroadcast_axes(output_tensor.shape, y.shape)

        output_tensor._unbroadcast_axis = [x_axis, y_axis]

        return output_tensor
    return _op


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





