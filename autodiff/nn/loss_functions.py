from autodiff.tensor import Tensor
from autodiff.utils import check, to_logits, clip_stable
import numpy as np


####### -------------- UNSTABLE BINARY CROSS ENTROPY LOSS --------------  ########
##################################################################################

class TensorBinaryCrossEntropy:
    """
    Unstable. Recommend using BinaryCrossEntropyLogits instead if using sigmoid
    function, or to just bypass the sigmoid fucntion and use SigmoidCrossEntropy
    instead.

    """
    def __call__(self, pred, actual):
        EPS = 1e-6

        a = pred * (actual + EPS) + (1 - pred) * (1 - actual + EPS)

        return -np.log(a)

# Alias
UnstableBinaryCrossEntropy = TensorBinaryCrossEntropy



####### -------------- CROSS ENTROPY LOSS --------------  ########
##################################################################

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



####### -------------- CATEGORICAL CROSS ENTROPY LOSS --------------  ########
##############################################################################

class CrossEntropy:
    def __call__(self, pred, actual):
        self.out = pred.categorical_cross_entropy_loss(actual)
        return self.out 

    def backward(self): self.out.backward()

# Aliases
CategoricalCrossEntropy = CrossEntropy
CCE = CrossEntropy
CCE_loss = CrossEntropy




####### -------------- MAE LOSS (L1) --------------  ########
#############################################################

####### -------------- MSE LOSS (L2) --------------  ########
#############################################################

class MSE:
    def __call__(self, pred, actual):
        return 0.5 * ((pred - actual) ** 2)

# Aliases
SquaredLoss = MSE
L2 = MSE



####### -------------- HINGE LOSS --------------  ########
##########################################################

####### -------------- Softmax Categorical Cross Entropy Bypass --------------  ########
########################################################################################

####### -------------- Sigmoid Binary Cross Entropy Bypass --------------  ########
###################################################################################









