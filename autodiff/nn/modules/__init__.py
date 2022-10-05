from .activation_functions import ReLU, Sigmoid, Softmax, LeakyReLU, Tanh
from .linear_layers import Linear
from .loss_functions import (
    SigmoidBinaryCrossEntropy,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SoftmaxCategoricalCrossEntropy,
    MSE
)


##### ----- ALIASES ----- #####

BCE = BinaryCrossEntropy

CrossEntropy = CategoricalCrossEntropy
CCE          = CategoricalCrossEntropy





