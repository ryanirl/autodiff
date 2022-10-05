from autodiff.nn.containers import Module 
from autodiff.tensor import Tensor

import numpy as np


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, bias = False):
        super().__init__()

        self.kernel = Tensor.uniform(out_channels, in_channels, kernel_size, kernel_size)

        self.kernel.stride  = stride
        self.kernel.padding = padding

        self.params.append(self.kernel)

        self.needs_bias = bias
        if self.needs_bias:
            self.bias = Tensor.uniform(out_channels)
            self.params.append(self.bias)

    def __call__(self, x):
        """
        Input Shape: (N, in_channels, H, W)
        Output Shape: (N, out_channels, H_out, W_out)

        """
        output = x.conv2d(self.kernel)

        if self.needs_bias: return output + self.bias
        else: return output





