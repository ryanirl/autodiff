

class SigmoidBinaryCrossEntropy:
    def __call__(self, pred, actual):
        return pred.sigmoid_binary_cross_entropy(actual)


class BinaryCrossEntropy:
    """ This using logits for numeric stability. """
    def __call__(self, pred, actual):
        return pred.stable_binary_cross_entropy_loss(actual)


class CategoricalCrossEntropy:
    def __call__(self, pred, actual):
        return pred.categorical_cross_entropy_loss(actual)


class SoftmaxCategoricalCrossEntropy:
    def __call__(self, pred, actual):
        return pred.softmax_categorical_cross_entropy(actual)


class MSE:
    def __call__(self, pred, actual):
        return 0.5 * ((pred - actual) ** 2)





