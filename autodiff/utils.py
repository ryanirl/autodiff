import numpy as np

def check(x, Type): 
    return x if isinstance(x, Type) else Type(x)

def primitive(Class):
    def register_methods(method):
        setattr(Class, method.__name__, method)
        return method 
    return register_methods

def _isscalar(tensor):
    if (tensor.shape == 1) or (tensor.shape == (1, )) or (tensor.shape == (1, 1)):
        return True
    else:
        return False

def to_logits(pred):
    EPS = 1e-06

    pred = np.clip(pred, EPS, 1.0 - EPS)

    logits = np.log(pred / (1.0 - pred))

    return logits

def unbroadcast_axes(shape_in, shape_out):
    # Return a tuple of axis to reduce along going
    # from shape_in -> shape_out

    # np.nonzero may be very slow | might consider using simple "for loop" for this.
    reduction_axes = np.nonzero((np.asarray(shape_out) < np.asarray(shape_in)) & (np.asarray(shape_out) == 1))[0]

    return tuple(reduction_axes)

def add_unbroadcast(grad, to_shape):
    sum_axes = unbroadcast_axes(np.shape(grad), to_shape)
    
    return np.reshape(np.sum(grad, axis = sum_axes, keepdims = True), to_shape)
