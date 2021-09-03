# Finally getting started on convolution layers

from autodiff.tensor import Tensor
from autodiff.nn.containers import Module 
import numpy as np


class Conv1D:
    def __init__(self):
        pass

class Conv2D:
    def __init__(self, channels, filters, kernel_size = 3, stride = 1, padding = 0, bias = True):
        """
        STILL NEED TO DO EXTENSIVE TESTING WITH THIS

        weights in shape: (filters, channels, kernel_size, kernel_size)
        x in shape: (N, channels, image_height, image_width)

        """
        self.stride = stride
        self.padding = padding 

        self.weights = Tensor.uniform(filters, channels, kernel_size, kernel_size)

        self.use_bias = bias
        if self.use_bias: self.bias = Tensor.uniform(filters)


    def __call__(self, x):
        """
        For now 'x' must be a tensor of size: (N, channels, height, width)

        """
        self.x = x

        if self.use_bias: return x.conv2d(self.weights, self.stride, self.padding) + self.bias
        else: return x.conv2d(self.weights, self.stride, self.padding)


