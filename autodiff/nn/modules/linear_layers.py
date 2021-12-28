from autodiff.tensor import Tensor
from autodiff.nn.containers import Module


class Linear(Module):
    def __init__(self, dims_in, dims_out, bias = True):
        super().__init__()

        self.weight = Tensor.uniform(dims_in, dims_out)
        self.params.append(self.weight)

        self.needs_bias = bias
        if self.needs_bias == True: 
            self.bias = Tensor.uniform(dims_out)
            self.params.append(self.bias)

    def __call__(self, X):
        output = X.dot(self.weight)

        if self.needs_bias: return output + self.bias
        else: return output





