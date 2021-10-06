from autodiff.tensor import primitive
from autodiff.nn.utils import to_logits, clip_stable, stable_bce

import numpy as np

### --- autodiff/nn/ops.py --- ###
#
# TODO:
#   - Conv2D Reimplement
# 
#
# NN OPS:
#   Activation Functions:
#       - Softmax
#
#   Loss Functions
#       - Categorical Cross Entropy (one-hot)
#       - Binary Cross Entropy
# 
#       Loss Function w/ Activation (bypass)
#           - Sigmoid Binary Cross Entropy
#               - EX: Logistic Regression
# 
#           - Softmax Categorical Cross Entropy
# 



### --- ACTIVATION FUNCTIONS --- ###  

class softmax(primitive):
    def forward(x):
        a = np.exp(x.value - np.max(x.value))

        return a / np.sum(a, axis = 1, keepdims = True)

    def backward(g, x, z):
        a = z.value[..., None] * z.value[:, None, :]
        b = np.einsum('ijk,ik->ij', a, g)

        return [g * z.value - b]


### --- LOSS FUNCTIONS --- ### 

class categorical_cross_entropy_loss(primitive):
    def forward(pred, actual):
        return -np.sum(actual.value * np.log(clip_stable(pred.value)), axis = 1, keepdims = True) 

    def backward(g, pred, actual, z):
        return [(-actual.value / (pred.value + 1e-6)), ]


class softmax_categorical_cross_entropy(primitive):
    def forward(pred, actual):
        return -np.sum(actual.value * np.log(clip_stable(softmax.forward(pred.value))), axis = 1, keepdims = True)

    def backward(g, pred, actual, z):
        return [g * (softmax.forward(pred.value) - actual.value), ]


# I learned about the stable BCE from here:
# https://rafayak.medium.com/how-do-tensorflow-and-keras-implement-binary-classification-and-the-binary-cross-entropy-function-e9413826da7
class stable_binary_cross_entropy_loss(primitive):
    def forward(pred, actual):
        pred.logit = to_logits(pred.value)
        return (1.0 / pred.shape[0]) * stable_bce(pred.logit, actual.value)

    def backward(g, pred, actual, z):
        return [g * ((1.0 / (1.0 + np.exp(-pred.logit))) - actual.value), ]


class sigmoid_binary_cross_entropy(primitive):
    def forward(pred, actual):
        return (1.0 / pred.shape[0]) * stable_bce(pred.value, actual.value)

    def backward(g, pred, actual, z):
        return [g * ((1.0 / (1.0 + np.exp(-pred.value))) - actual.value), ]



## 'col2im_numpy', '_conv2d_forward', and '_conv2d_backward' is a very lightly
## modified version from cs231n's fast_layers.py which can be found here:
## https://cs231n.github.io/assignments2021/assignment2/
#def col2im_numpy(cols, x_shape, filter_height = 3, filter_width = 3, padding = 1, stride = 1):
#    N, C, H, W = x_shape
#
#    out_h = int((H + 2 * padding - filter_height) / stride) + 1
#    out_w = int((W + 2 * padding - filter_width) / stride) + 1
#
#    H_padded, W_padded = H + 2 * padding, W + 2 * padding
#
#    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
#
#
#    i0 = np.tile(np.repeat(np.arange(filter_height), filter_width), C)
#    i1 = stride * np.repeat(np.arange(out_h), out_w)
#
#    j0 = np.tile(np.arange(filter_width), filter_height * C)
#    j1 = stride * np.tile(np.arange(out_w), out_h)
#
#    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
#    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
#
#    k = np.repeat(np.arange(C), filter_height * filter_width).reshape(-1, 1)
#
#    cols_reshaped = cols.reshape(C * filter_height * filter_width, -1, N)
#
#    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
#
#    # This is what takes the most time and what I will be looking to optimize
#    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped) 
#
#    if padding == 0: return x_padded
#    return x_padded[:, :, padding : -padding, padding : -padding]
#
#def _conv2d_forward(x, weights):
#    N, C, H, W = x.shape
#    F, _, FH, FW = weights.shape
#    stride, pad = x.stride, x.padding 
#
#    x_padded = np.pad(x.value, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode = "constant")
#
#    H = H + (2 * pad)
#    W = W + (2 * pad)
#
#    out_h = int((H - FH) / stride) + 1
#    out_w = int((W - FW) / stride) + 1
#
#    shape = (C, FH, FW, N, out_h, out_w)
#
#    strides = x.value.itemsize * np.array((H * W, W, 1, C * H * W, stride * W, stride))
#
#    im2col = np.ascontiguousarray(as_strided(x_padded, shape = shape, strides = strides))
#
#    im2col.shape = (C * FH * FW, N * out_h * out_w) 
#
#    x.cached_im2col = im2col
#
#    out = weights.reshape(F, -1).dot(im2col)
#
#    out.shape = (F, N, out_h, out_w)
#
#    return np.ascontiguousarray(out.transpose(1, 0, 2, 3)) 
#
#
#def _conv2d_backward(ingrad, x, weights, z):
#    N, C, H, W = x.shape
#    F, _, FH, FW = weights.shape
#    stride, pad = x.stride, x.padding 
#    _, _, out_h, out_w = ingrad.shape
#
#    ingrad = ingrad.transpose(1, 0, 2, 3).reshape(F, -1)
#
#    dw = ingrad.dot(x.cached_im2col.T).reshape(weights.shape)
#
#    dx_im2col = weights.value.reshape(F, -1).T.dot(ingrad)
#    dx_im2col.shape = (C, FH, FW, N, out_h, out_w)
#
##    dx = col2im(dx_im2col, (N, C, H, W), FH, FW, pad, stride)
#    dx = col2im_numpy(dx_im2col, (N, C, H, W), FH, FW, pad, stride)
#
#    return [dx, dw]



### --- REGISTER OPS --- ###

softmax_categorical_cross_entropy()
stable_binary_cross_entropy_loss()
categorical_cross_entropy_loss()
sigmoid_binary_cross_entropy()
softmax()





