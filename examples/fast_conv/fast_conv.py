# If you were looking to speed up the Conv backward pass
# this will provide everything you need to do it using
# ctypes which is a defauly Python library.

import ctypes

# C binded backwards pass
def col2im(cols, x_shape, filter_height = 3, filter_width = 3, padding = 1, stride = 1):
    N, C, H, W = x_shape

    H_padded, W_padded = H + 2 * padding, W + 2 * padding

    cols.dtype = "double"

    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    x_padded_shape = x_padded.shape

    libc = ctypes.CDLL("./autodiff/_col2im.so")

    pointer1 = cols.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pointer2 = x_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    out = libc.col2im_cpu(pointer1, pointer2, N, C, H_padded, W_padded, filter_height, filter_width, stride, padding)

    x_padded = np.ctypeslib.as_array(pointer2, shape = x_padded_shape).copy()

    if padding == 0: return x_padded
    return x_padded[:, :, padding : -padding, padding : -padding]


