from autodiff.tensor import Tensor
from autodiff.utils import check, to_logits, clip_stable
import numpy as np


class TensorBinaryCrossEntropy:
    """
    This is UNSTABLE. Recommend using BinaryCrossEntropyLogits instead if using
    sigmoid function, or to just bypass the sigmoid fucntion and use
    SigmoidBinaryCrossEntropy instead to save computation time.

    """
    def __call__(self, pred, actual):
        a = pred * (actual + 1e-6) + (1 - pred) * (1 - actual + 1e-6)

        return -np.log(a)



class SigmoidBinaryCrossEntropy:
    """
    This actual does the divide too. I need to fix this for the other loss
    functions.
    
    This is like a loss and activation fucntion at the same time.

    """
    def __call__(self, pred, actual):
        self.pred = pred

        return self.pred.sigmoid_binary_cross_entropy(actual)

    def backward(self):
        return self.pred.backward()



class BinaryCrossEntropy:
    """
    This using logits for numeric stability.

    """
    def __call__(self, pred, actual):
        self.out = pred.stable_binary_cross_entropy_loss(actual)
        return self.out

    def backward(self): self.out.backward()



class CategoricalCrossEntropy:
    def __call__(self, pred, actual):
        self.out = pred.categorical_cross_entropy_loss(actual)
        return self.out 

    def backward(self): self.out.backward()



class SoftmaxCategoricalCrossEntropy:
    def __call__(self, pred, actual):
        self.pred = pred

        return self.pred.softmax_categorical_cross_entropy(actual)

    def backward(self):
        return self.pred.backward()



class MSE:
    def __call__(self, pred, actual):
        return 0.5 * ((pred - actual) ** 2)



####### -------------- MAE LOSS (L1) --------------  ########
#############################################################

####### -------------- HINGE LOSS --------------  ########
##########################################################



##### ----- ALIAS ----- #####

UnstableBinaryCrossEntropy = TensorBinaryCrossEntropy

BinaryCrossEntropyLogits = BinaryCrossEntropy
BinaryCrossEntropy = BinaryCrossEntropy
StableBinaryCrossEntropy = BinaryCrossEntropy
BCE = BinaryCrossEntropy
BinaryCE = BinaryCrossEntropy

CrossEntropy = CategoricalCrossEntropy
CCE = CategoricalCrossEntropy
Categorical_CE = CategoricalCrossEntropy

SquaredLoss = MSE
L2 = MSE





