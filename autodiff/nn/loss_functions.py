
class SigmoidBinaryCrossEntropy:
    """
    This actual does the divide too. I need to fix this for the other loss
    functions.
    
    This is like a loss and activation fucntion at the same time.

    """
    def __call__(self, pred, actual):
        self.loss = pred.sigmoid_binary_cross_entropy(actual)

        return self.loss

    def backward(self):
        self.loss.backward()


class BinaryCrossEntropy:
    """
    This using logits for numeric stability.

    """
    def __call__(self, pred, actual):
        self.loss = pred.stable_binary_cross_entropy_loss(actual)

        return self.loss

    def backward(self): 
        self.loss.backward()


class CategoricalCrossEntropy:
    def __call__(self, pred, actual):
        self.loss = pred.categorical_cross_entropy_loss(actual)

        return self.loss

    def backward(self): 
        self.loss.backward()


class SoftmaxCategoricalCrossEntropy:
    def __call__(self, pred, actual):
        self.loss = pred.softmax_categorical_cross_entropy(actual)

        return self.loss

    def backward(self):
        self.loss.backward()


class MSE:
    def __call__(self, pred, actual):
        return 0.5 * ((pred - actual) ** 2)


# TODO:
    # MAE Loss (L1)
    # Hinge Loss




##### ----- ALIASES ----- #####

BinaryCrossEntropyLogits = BinaryCrossEntropy
BinaryCrossEntropy       = BinaryCrossEntropy
StableBinaryCrossEntropy = BinaryCrossEntropy
BCE                      = BinaryCrossEntropy
BinaryCE                 = BinaryCrossEntropy


CrossEntropy   = CategoricalCrossEntropy
CCE            = CategoricalCrossEntropy
Categorical_CE = CategoricalCrossEntropy


SquaredLoss = MSE
L2          = MSE





