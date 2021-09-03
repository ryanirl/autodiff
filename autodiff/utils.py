import numpy as np
from numpy.lib.stride_tricks import as_strided

### --- Tensor Utils --- ###
def check(x, Type): 
    return x if isinstance(x, Type) else Type(x)


def primitive(Class):
    def register_methods(method):
        setattr(Class, method.__name__, method) 
        return method 
    return register_methods 

### --- Numerically Stable Utils --- ###

def clip_stable(value):
    EPS = 1e-6
    return np.clip(value, EPS, 1.0 - EPS)


def to_logits(pred):
    EPS = 1e-06

    pred = np.clip(pred, EPS, 1.0 - EPS)

    logits = np.log(pred / (1.0 - pred))

    return logits


### --- Broadcasting Utils --- ###

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


### --- Convolution Utils --- ###

# 'col2im_numpy', '_conv2d_forward', and '_conv2d_backward' is a very lightly
# modified version from cs231n's fast_layers.py which can be found here:
# https://cs231n.github.io/assignments2021/assignment2/
def col2im_numpy(cols, x_shape, filter_height = 3, filter_width = 3, padding = 1, stride = 1):
    N, C, H, W = x_shape

    out_h = int((H + 2 * padding - filter_height) / stride) + 1
    out_w = int((W + 2 * padding - filter_width) / stride) + 1

    H_padded, W_padded = H + 2 * padding, W + 2 * padding

    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)


    i0 = np.tile(np.repeat(np.arange(filter_height), filter_width), C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)

    j0 = np.tile(np.arange(filter_width), filter_height * C)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), filter_height * filter_width).reshape(-1, 1)

    cols_reshaped = cols.reshape(C * filter_height * filter_width, -1, N)

    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # This is what takes the most time and what I will be looking to optimize
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped) 

    if padding == 0: return x_padded
    return x_padded[:, :, padding : -padding, padding : -padding]

def _conv2d_forward(x, weights):
    N, C, H, W = x.shape
    F, _, FH, FW = weights.shape
    stride, pad = x.stride, x.padding 

    x_padded = np.pad(x.value, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode = "constant")

    H = H + (2 * pad)
    W = W + (2 * pad)

    out_h = int((H - FH) / stride) + 1
    out_w = int((W - FW) / stride) + 1

    shape = (C, FH, FW, N, out_h, out_w)

    strides = x.value.itemsize * np.array((H * W, W, 1, C * H * W, stride * W, stride))

    im2col = np.ascontiguousarray(as_strided(x_padded, shape = shape, strides = strides))

    im2col.shape = (C * FH * FW, N * out_h * out_w) 

    x.cached_im2col = im2col

    out = weights.reshape(F, -1).dot(im2col)

    out.shape = (F, N, out_h, out_w)

    return np.ascontiguousarray(out.transpose(1, 0, 2, 3)) 


def _conv2d_backward(ingrad, x, weights, z):
    N, C, H, W = x.shape
    F, _, FH, FW = weights.shape
    stride, pad = x.stride, x.padding 
    _, _, out_h, out_w = ingrad.shape

    ingrad = ingrad.transpose(1, 0, 2, 3).reshape(F, -1)

    dw = ingrad.dot(x.cached_im2col.T).reshape(weights.shape)

    dx_im2col = weights.value.reshape(F, -1).T.dot(ingrad)
    dx_im2col.shape = (C, FH, FW, N, out_h, out_w)

#    dx = col2im(dx_im2col, (N, C, H, W), FH, FW, pad, stride)
    dx = col2im_numpy(dx_im2col, (N, C, H, W), FH, FW, pad, stride)

    return [dx, dw]



