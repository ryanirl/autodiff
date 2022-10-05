import numpy as np
import ctypes


def col2im(cols, N, C, H, W, filter_height = 3, filter_width = 3, padding = 1, stride = 1):
    H_padded, W_padded = H + 2 * padding, W + 2 * padding

    cols.dtype = "double"
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype = cols.dtype)

    libc = ctypes.CDLL("./col2im_c/_col2im.cpython-39-darwin.so")

    pointer1 =     cols.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pointer2 = x_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # maybe h_padded is correct but then the out_shape becomes wrong.
    out = libc.col2im_cpu(pointer1, pointer2, N, C, H_padded, W_padded, filter_height, filter_width, stride, padding)

    x_padded = np.ctypeslib.as_array(pointer2, shape = x_padded.shape).copy()

    if padding == 0: return x_padded
    return x_padded[:, :, padding : -padding, padding : -padding]





