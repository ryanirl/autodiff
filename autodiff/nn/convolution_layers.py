# Finally getting started on convolution layers

from autodiff.tensor import Tensor
from autodiff.nn.containers import Module 
import numpy as np

# Pooling notes:
#    - Takes volume of size WxHxD
#    - 3 hyperparameters: Spatial Entent F, Stride S
#    - Producted a new W2xH2xD2
#        - W2 = ((W - F) / S) + 1
#        - H2 = ((H - F) / S) + 1
#        - D2 = D1
# Common Settings: F = 2, S = 2  ||| F = 2, S = 2

# Conv notes:
    # Takes in volume of size: W1xH1xD1
    # Hyperparameters: 
        # Number of filters K
        # Spacial extent F
        # Stride S
        # Amount of zero padding P

    # Returns volume of size W2xH2xD2 where:
        # W2 = ((W1 - F + 2P) / S) + 1
        # H2 = ((H1 - F + 2P) / S) + 1 
        # D2 = K

    
# Start with 2D everything

 
# Conv (Hyperparams: kernal type, kernal size, padding, stride)
# ReLU
# Pool (Hyperparams:?)
# Repeat

# MLP 
# Softmax
# CategoricalCE

# Predictions

class Conv1D:
    def __init__(self):
        pass


class Conv2D:
    """
    NOT WORKING YET, base functionallity has been layed

    Till I update functionality is limited to:
        - No padding
        - Stride 1
        - Only 1 channel works
        - Square image
        - Square kernel / weight
        - No bias
        - No groups

    'filter_size' & 'stride' must be integers

    What about FFT's?

    """
    def __init__(self, x, actiavation_maps, kernel_size = 3, stride = 1, bias = True):
        self.x = np.atleast_3d(x) # X must be a square
        self.xm, self.xn, self.depth = self.x.shape
        self.activation_maps = activation_maps

        self.filter_size = filter_size 
        self.filter_shape = np.array([filter_size, filter_size, self.depth, activation_maps]) 

        self.stride = stride
        self.stride_dim = np.array([stride, stride])

        self.outH = int(((self.xm - filter_size) / stride) + 1)
        self.outW = int(((self.xn - filter_size) / stride) + 1)

        # Later add activation maps???
        self.out_shape = np.array([self.outH, self.outW, self.depth])

        self.weight = Tensor.uniform(filter_size, filter_size, self.depth, activation_maps)
        if bias: self.bias = Tensor.uniform(activation_maps)

    def forward(self):
        """
        NEED TO MOVE THIS AS TENSOR OP

        """
        im2col_shape = [*self.out_shape, self.filter_size, self.filter_size, self.activation_maps]

        strides = np.array([*self.x.strides, *self.x.strides])

        im2col = np.lib.stride_tricks.as_strided(self.x, shape = im2col_shape, strides = strides, writeable = False)

        # This seems to be a little slower than I would like, though still pretty darn quick compared to naive version
        # I believe mem strided im2col with numpy + einsum gave about 100-150x increase.
        activation_map = np.einsum('xyzijk,ijzk -> xyzk',im2col, self.filter) # Fully vectorized version
#        activation_map = np.einsum('xyzijk,ijzk -> xyz',im2col, self.filter) # Vectorized only for single act_map

        return activation_map

    def backward(self):
        pass


class ConvTranspose1D:
    def __init__(self):
        pass

class ConvTranspose2D:
    def __init__(self):
        pass

