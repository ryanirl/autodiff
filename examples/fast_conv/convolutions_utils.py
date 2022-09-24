import numpy as np

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


def conv2d_output_shape(H, W, kernel_size, padding, stride, out_channels):
    H_out = (H - kernel_size[0] + (2 * padding)) / stride[0]
    W_out = (W - kernel_size[1] + (2 * padding)) / stride[1]

    return [H_out, W_out, out_channels]





