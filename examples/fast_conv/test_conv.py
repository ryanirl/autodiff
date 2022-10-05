import sys
sys.path.insert(0, "../../")
from autodiff.tensor import Tensor
import convolutions_functional 

import numpy as np
import torch


def torch_compare(x, stride = 1, kernel_size = 3, padding = 1):
    """ 
    Given one of the above functions, this will compare the forward and
    backwards pass with the PyTorch implementation.

    Must have PyTorch installed.

    """
    import torch
    import torch.nn.functional as F

    kernel_torch = torch.rand((1, 3, 5, 5), dtype = torch.float32, requires_grad = True)
    kernel_custom = Tensor(kernel_torch.clone().detach().numpy(), requires_grad = True)

    # Make sure the types are the same.
#    print(kernel_torch.dtype)
#    print(kernel_custom.value.dtype)

    # Generate the torch and autodiff specific Tensors.
    x_torch  = x.float().requires_grad_(True)

    x_custom = Tensor(x.clone().detach().numpy().astype(np.float32), requires_grad = True)
    x_custom.stride  = stride
    x_custom.padding = padding
    kernel_custom.stride  = stride
    kernel_custom.padding = padding

    # Generate the random kernel of some predefined size.
    out_torch  = F.conv2d(x_torch, kernel_torch,  bias = None, stride = stride, padding = padding)

    out_custom = x_custom.conv2d(kernel_custom)
    out_custom.backward()

    # Then compare the two. Need to do forward and backward testing.
    # Taken from my bias.py file:

    # First compare the forward pass.
    # To make sure the arrays are equal within "some threshold" we use np.allclose().
    for_eq = np.allclose(out_custom.value, out_torch.detach().numpy())
    print(f"Custom shape: {out_custom.value.shape} | Torch shape: {out_torch.shape}")
    print(f"Forward pass: {for_eq}")

    # First compare the backward pass.
    # NOTES: 
    # - Need to check the x and kernel grad.
    x_grad_torch, kernel_grad_torch = torch.autograd.grad(out_torch, [x_torch, kernel_torch], torch.Tensor(np.ones(out_torch.shape)))
    x_grad_torch      = x_grad_torch.detach().numpy()
    kernel_grad_torch = kernel_grad_torch.detach().numpy()

    x_grad_custom      = x_custom.grad
    kernel_grad_custom = kernel_custom.grad
#    print(x_grad_custom.shape)
#    print(x_grad_torch.shape)
#    print(x_grad_custom)
#    print(x_grad_torch)
#    print(kernel_grad_custom.shape)
#    print(kernel_grad_torch.shape)
    
    back_x_eq      = np.allclose(x_grad_torch, x_grad_custom)
    back_kernel_eq = np.allclose(kernel_grad_torch, kernel_grad_custom)
    print(f"Backward pass KERNEL: {back_kernel_eq}")
    print(f"Backward pass X     : {back_x_eq}")
    
if __name__ == "__main__":
    # 1. Gather the 'x' data.
    # 2. Test each forward and backward.

    x = torch.randint(low = 0, high = 255, size = (1, 3, 100, 100))

    torch_compare(x)




