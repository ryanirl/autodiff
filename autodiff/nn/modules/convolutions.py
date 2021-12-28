from autodiff.nn.containers import Module 
from autodiff.tensor import Tensor
import numpy as np


class Conv1D:
    def __init__(self):
        pass

    def __call__(self):
        pass


class Conv2D:
    def __init__(self, channels_in, channels_out, kernel_size = 3, stride = 1, padding = 0, bias = False):
        self.weights = Tensor.uniform(channels_out, channels_in, kernel_size, kernel_size)

    def __call__(self, x):
        """
        Input Shape: (N, in_channels, H, W)
        Output Shape: (N, out_channels, H_out, W_out)

        """
        pass


class Conv3D:
    def __init__(self):
        pass

    def __call__(self):
        pass

