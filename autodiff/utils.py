import numpy as np

def check(x, Type): 
    return x if isinstance(x, Type) else Type(x)


def clip_stable(value):
    EPS = 1e-6
    return np.clip(value, EPS, 1.0 - EPS)


def primitive(Class):
    def register_methods(method):
        setattr(Class, method.__name__, method)
        return method 
    return register_methods


def to_logits(pred):
    EPS = 1e-06

    pred = np.clip(pred, EPS, 1.0 - EPS)

    logits = np.log(pred / (1.0 - pred))

    return logits


def unbroadcast_axes(shape_in, shape_out):
    """
    Optimizing this is a huge problem I'm currently working on.
    Using numpy np.where() function is extremely slow. A for loop
    works but there must be some really quick solution I haven't
    thought about.

    Idea: Given an ingrad with different shape from the gradient 
    of our current value we must re-shape the ingrad to match the
    shape of our current gradient.

    ---

    Given: shape_in: (3, 2) & shape_out: (3, 1)
    
    Sum along axis (1) bec 2 -> 1 at index 1.

    ---

    Given shape_in: (3, 4, 10, 5, 3) & shape_out: (1, 4, 1, 1, 3)

    Sum along axis (0, 2, 3) because index's 0, 2, and 3 shrunk to 1

    """

    if shape_out == (): return None

    # This numpy fucntion is very slow. Don't recommend using.
    # reduction_axes = np.nonzero((np.asarray(shape_out) < np.asarray(shape_in)) & (np.asarray(shape_out) == 1))[0]

    reduction_axes = []

    # There must be something simplier than this.
    for i in range(len(shape_in)):
        if (shape_in[i] > shape_out[i]) & (shape_out[i] == 1):
            reduction_axes += [i]

    return tuple(reduction_axes)


def _unbroadcast(grad, to_shape):
    sum_axes = unbroadcast_axes(np.shape(grad), to_shape)

    return np.reshape(np.sum(grad, axis = sum_axes, keepdims = True), to_shape)






