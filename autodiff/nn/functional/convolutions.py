from autodiff.nn.utils import to_logits, clip_stable, stable_bce
from autodiff.tensor import register

import numpy as np


@register
class conv1d:
    def forward(x, kernel):
        return

    def backward(g, x, kernel, output):
        return 


@register
class conv2d:
    def forward(x, kernel):
        return

    def backward(g, x, kernel, output):
        return 


@register
class conv3d:
    def forward(x, kernel):
        return

    def backward(g, x, kernel, output):
        return 



