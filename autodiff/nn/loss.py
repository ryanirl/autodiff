from autodiff.tensor import Tensor, OP
from autodiff.ops import grad_fun, value_fun
from autodiff.utils import primitive, check, to_logits
import numpy as np


####### -------------- UNSTABLE BINARY CROSS ENTROPY LOSS --------------  ########

class TensorBinaryCrossEntropy:
    """
    Unstable

    """
    def __call__(self, pred, actual):
        EPS = 1e-6

        a = pred * (actual + eps) + (1 - pred) * (1 - actual + eps)

        return -(np.log(a))

# Alias
UnstableBinaryCrossEntropy = TensorBinaryCrossEntropy

##################################################################################


####### -------------- CROSS ENTROPY LOSS --------------  ########

def stable_binary_cross_entropy_forward(pred, actual):
    pred = to_logits(pred)

    out = 0.01 * np.sum(np.maximum(pred, 0) - (pred * actual) + np.log(1.0 + np.exp(-np.abs(pred))))

    return out 

value_fun["stable_binary_cross_entropy_loss"] = (lambda x, y: stable_binary_cross_entropy_forward(x.value, y.value))
grad_fun["stable_binary_cross_entropy_loss"] = (lambda g, pred, actual, z: (g * ((1.0 / (1.0 + np.exp(-to_logits(pred.value)))) - actual.value), ))

@primitive(Tensor)
def stable_binary_cross_entropy_loss(self, actual):
    return OP("stable_binary_cross_entropy_loss", self, actual);

class StableBinaryCrossEntropy:
    def __call__(self, pred, actual):
        self.out = pred.stable_binary_cross_entropy_loss(actual)
        return self.out

    def backward(self): self.out.backward()

# Aliases
BinaryCrossEntropyLogits = StableBinaryCrossEntropy
BinaryCrossEntropy = StableBinaryCrossEntropy
BCE = StableBinaryCrossEntropy
BinaryCE = StableBinaryCrossEntropy

##################################################################


####### -------------- CATEGORICAL CROSS ENTROPY LOSS --------------  ########

value_fun["cross_entropy_loss"] = (lambda pred, actual: -np.sum(actual.value * np.log(pred.value), axis = 1, keepdims = True))

grad_fun["cross_entropy_loss"] = (lambda g, pred, actual, z: (g * (-actual.value / (pred.value + 1e-6)), ))

@primitive(Tensor)
def cross_entropy_loss(self, actual):
    return OP("cross_entropy_loss", self, actual);

class CrossEntropy:
    def __call__(self, pred, actual):
        self.out = pred.cross_entropy_loss(actual)
        return self.out

    def backward(self): self.out.backward()

# Aliases
CategoricalCrossEntropy = CrossEntropy
CCE = CrossEntropy
CCE_loss = CrossEntropy

##################################################################



####### -------------- MAE LOSS (L1) --------------  ########
#############################################################


####### -------------- MSE LOSS (L2) --------------  ########

class MSE:
    def __call__(self, pred, actual):
        return 0.5 * ((pred - actual) ** 2)

# Aliases
SquaredLoss = MSE
L2 = MSE

#############################################################


####### -------------- HINGE LOSS --------------  ########
##########################################################











