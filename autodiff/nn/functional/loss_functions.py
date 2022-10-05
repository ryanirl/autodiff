# References:
# ------------
# Stable BCE: https://rafayak.medium.com/how-do-tensorflow-and-keras-implement-binary-classification-and-the-binary-cross-entropy-function-e9413826da7
# ------------
from autodiff.nn.utils import to_logits, clip_stable, stable_bce
from autodiff.tensor import register

import numpy as np


@register
class categorical_cross_entropy_loss:
    def forward(pred, actual):
        a = np.log(clip_stable(pred.value))

        return -np.sum(actual.value * a, axis = 1, keepdims = True) 

    def backward(g, pred, actual, z):
        g_pred = -actual.value / (pred.value + 1e-6)

        return [g_pred, ]


@register
class softmax_categorical_cross_entropy:
    def forward(pred, actual):
        a = np.log(clip_stable(softmax.forward(pred.value)))

        return -np.sum(actual.value * a, axis = 1, keepdims = True)

    def backward(g, pred, actual, z):
        g_pred = g * (softmax.forward(pred.value) - actual.value)

        return [g_pred, ]


@register
class stable_binary_cross_entropy_loss:
    def forward(pred, actual):
        pred.logit = to_logits(pred.value)
        return (1.0 / pred.shape[0]) * stable_bce(pred.logit, actual.value)

    def backward(g, pred, actual, z):
        g_pred = g * ((1.0 / (1.0 + np.exp(-pred.logit))) - actual.value)

        return [g_pred, ]


@register
class sigmoid_binary_cross_entropy:
    def forward(pred, actual):
        return (1.0 / pred.shape[0]) * stable_bce(pred.value, actual.value)

    def backward(g, pred, actual, z):
        g_pred = g * ((1.0 / (1.0 + np.exp(-pred.value))) - actual.value)

        return [g_pred, ]





