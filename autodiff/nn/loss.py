from autodiff.tensor import Tensor, OP
from autodiff.ops import grad_fun, value_fun
from autodiff.utils import primitive, check
import numpy as np


class TensorBinaryCrossEntropy:
    def __call__(self, pred, actual):
        return (-actual) * pred.log() + ((1 - actual) * (1 - pred).log())




####### -------------- CROSS ENTROPY LOSS --------------  ########

def cross_entropy_forward(pred, actual):
    out = -np.sum(actual * np.log(pred), axis = 1, keepdims = True)

    return out

def cross_entropy_backward(pred, actual):
    eps = 1e-6

    return -actual / (pred + eps) 

value_fun["cross_entropy_loss"] = (lambda x, y: cross_entropy_forward(x.value, y.value))

grad_fun["cross_entropy_loss"] = (lambda g, x, y, z: ((cross_entropy_backward(x.value, y.value)), ))

@primitive(Tensor)
def cross_entropy_loss(self, actual):
    return OP("cross_entropy_loss", self, actual);

class CrossEntropy:
    def __call__(self, pred, actual):
        # PRED MUST BE TENSOR
        self.out = pred.cross_entropy_loss(actual)
        return self.out

    def backward(self): self.out.backward()

##################################################################



####### -------------- MAE LOSS (L1) --------------  ########
#############################################################

####### -------------- MSE LOSS (L2) --------------  ########
#############################################################

####### -------------- HINGE LOSS --------------  ########
##########################################################











