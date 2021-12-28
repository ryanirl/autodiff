from autodiff.nn.utils import to_logits, clip_stable, stable_bce
from autodiff.tensor import register

import numpy as np


@register
class MaxPool1D:
    def forward(x):
        pass

    def backward(g, x, z):
        pass


@register
class MaxPool2D:
    def forward(x):
        pass

    def backward(g, x, z):
        pass


@register
class MaxPool3D:
    def forward(x):
        pass

    def backward(g, x, z):
        pass


@register
class AvgPool1D:
    def forward(x):
        pass

    def backward(g, x, z):
        pass


@register
class AvgPool2D:
    def forward(x):
        pass

    def backward(g, x, z):
        pass


@register
class AvgPool3D:
    def forward(x):
        pass

    def backward(g, x, z):
        pass
