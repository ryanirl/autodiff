# THESE WE IN TENSOR
from autodiff.utils import primitive
from autodiff.tensor import Tensor, OP

#    def tensor_softmax(self):
#        a = (self - self.max()).exp()
#        b = a.sum(axis = 1, keepdims = True)
#
#        return a / b


#class Softmax:
#    @staticmethod
#    def forward(self, x):
#        pass
#
#    @staticmethod
#    def backward(self, g, x, z):
#        pass
#
#
#
#@primitive(Tensor)
#def softmax(self):
#    return OP("softmax", self)

#    ### --- Loss Functions --- ###
##
#
#@primitive(Tensor)
#def stable_binary_cross_entropy_loss(self, actual):
#    return OP("stable_binary_cross_entropy_loss", self, actual)
##
#@primitive(Tensor)
#def categorical_cross_entropy_loss(self, actual):
#    return OP("categorical_cross_entropy_loss", self, actual)
##
#@primitive(Tensor)
#def sigmoid_binary_cross_entropy(self, actual):
#    return OP("sigmoid_binary_cross_entropy", self, actual)
##
#    def softmax_categorical_cross_entropy(self, actual):
#        return OP("softmax_categorical_cross_entropy", self, actual)
#

    ### --- Convolutions --- ###

#    def conv2d(self, weight, stride, padding):
#        self.stride = stride
#        self.padding = padding
#        self.cached_im2col = None
#
#        return OP("conv2d", self, check(weight, Tensor))

#    def pool2d(self, pool_stride, pool_filter_size):
#        self.pool_stride = pool_stride
#        self.pool_filter_size = pool_filter_size



# put nn utils here :)
import numpy as np
from numpy.lib.stride_tricks import as_strided

### --- Numerically Stable Utils --- ###

def clip_stable(value):
    EPS = 1e-6
    return np.clip(value, EPS, 1.0 - EPS)


def to_logits(pred):
    EPS = 1e-06

    pred = np.clip(pred, EPS, 1.0 - EPS)

    logits = np.log(pred / (1.0 - pred))

    return logits


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



