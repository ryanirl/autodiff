from autodiff.tensor import Tensor, OP
from autodiff.utils import check
import autodiff.nn as nn

import numpy as np
import time 


def test_grad():
    # Should be:
    #  - Value: 22500
    #  - df/dA: 310500
    #  - df/dB: 242996
    #  - df/dC: -3

    a = Tensor(2)
    b = Tensor(3)
    c = Tensor(4)

    z = a + b
    y = z ** 2
    x = y * b
    w = x * a
    v = w ** 2
    u = a * b
    t = v * u
    s = c * b
    r = t - s

    r.backward()

    print("value: ({}) should be 134988".format(r.value))
    print("grad wrt A: ({}) should be 310500".format(a.grad))
    print("grad wrt B: ({}) should be 242996".format(b.grad))
    print("grad wrt C: ({}) should be -3".format(c.grad))


def test_softmax():
    a = Tensor(np.array([[1, 2, 3], [1, 2, 3]]))
    b = Tensor(np.array([[0, 1, 0], [1, 0, 0]]))

    loss_func = nn.CrossEntropy() 

    c = a.softmax()

    loss = loss_func(c, b)

    loss.backward()

    print(c.grad)
    print(a.grad)


if __name__ == "__main__":
    test_grad()
    test_softmax()





