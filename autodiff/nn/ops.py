from autodiff.tensor import Tensor, primitive
from autodiff.nn.utils import to_logits, clip_stable
from autodiff.nn.utils import _conv2d_forward, _conv2d_backward

from collections import defaultdict
import numpy as np


class softmax(primitive):
    def forward(x):
        a = np.exp(x.value - np.max(x.value))

        return a / np.sum(a, axis = 1, keepdims = True)

    def backward(g, x, z):
        a = z.value[..., None] * z.value[:, None, :]
        b = np.einsum('ijk,ik->ij', a, g)

        return [g * z.value - b]


class categorical_cross_entropy_loss(primitive):
    def forward(pred, actual):
        return -np.sum(actual.value * np.log(clip_stable(pred.value)), axis = 1, keepdims = True) 

    def backward(g, pred, actual, z):
        return [(-actual.value / (pred.value + 1e-6)), ]


class stable_binary_cross_entropy_loss(primitive):
    def forward(pred, actual):
        pred = to_logits(pred.value) 
        return np.sum(np.maximum(pred, 0) - (pred * actual.value) + np.log(1.0 + np.exp(-np.abs(pred))))

    def backward(g, pred, actual, z):
        return [g * ((1.0 / (1.0 + np.exp(-to_logits(pred.value)))) - actual.value), ]


class softmax_categorical_cross_entropy(primitive):
    def forward(pred, actual):
        return -np.sum(actual.value * np.log(clip_stable(softmax.forward(pred.value))), axis = 1, keepdims = True)

    def backward(g, pred, actual, z):
        return [g * (softmax.forward(pred.value) - actual.value), ]



class sigmoid_binary_cross_entropy(primitive):
    def forward(pred, actual):
        return (1.0 / pred.shape[0]) * np.sum(np.maximum(pred.value, 0) - (pred.value * actual.value) + np.log(1.0 + np.exp(-np.abs(pred.value))))

    def backward(g, pred, actual, z):
        return [g * ((1.0 / (1.0 + np.exp(-pred.value))) - actual.value), ]




# REGISTER 
softmax_categorical_cross_entropy()
stable_binary_cross_entropy_loss()
categorical_cross_entropy_loss()
sigmoid_binary_cross_entropy()
softmax()





