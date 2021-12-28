from autodiff.nn.utils import to_logits, clip_stable, stable_bce
from autodiff.tensor import register

import numpy as np


@register
class softmax:
    def forward(x):
        a = np.exp(x.value - np.max(x.value))

        return a / np.sum(a, axis = 1, keepdims = True)

    def backward(g, x, z):
        a = z.value[..., None] * z.value[:, None, :]
        b = np.einsum('ijk,ik->ij', a, g)

        return [g * z.value - b]




