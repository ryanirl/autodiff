from autodiff.nn.utils import to_logits, clip_stable, stable_bce
from autodiff.nn.functional.utils import col2im_numpy
from autodiff.tensor import register

from numpy.lib.stride_tricks import as_strided
import numpy as np

from autodiff.nn.functional.cython_col2im import col2im_6d_cython


@register
class conv1d:
    def forward(x, kernel):
        return

    def backward(g, x, kernel, output):
        return 


@register
class conv2d:
    def forward(x, kernel):
        N, C, H, W = x.shape
        F, _, FH, FW = kernel.shape

        stride, pad = kernel.stride, kernel.padding

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

        out = kernel.value.reshape(F, -1).dot(im2col)

        out.shape = (F, N, out_h, out_w)

        return np.ascontiguousarray(out.transpose(1, 0, 2, 3)) 

    def backward(g, x, kernel, output):
        N, C, H, W = x.shape
        F, _, FH, FW = kernel.shape

        stride, pad = kernel.stride, kernel.padding 

        _, _, out_h, out_w = g.shape

        ingrad = g.transpose(1, 0, 2, 3).reshape(F, -1)

        dw = ingrad.dot(x.cached_im2col.T).reshape(kernel.shape)

        dx_im2col = kernel.value.reshape(F, -1).T.dot(ingrad)
        dx_im2col.shape = (C, FH, FW, N, out_h, out_w)

        dx = col2im_6d_cython(dx_im2col, N, C, H, W, FH, FW, pad, stride)

        return [dx, dw]

@register
class testing:
    def forward(x, y):
        return x.value - y.value

    def backward(g, x, y, z):
        return [g, -g]





