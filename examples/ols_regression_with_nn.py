# This is by far the simplest example I could come up with
from autodiff.tensor import Tensor
import autodiff.nn as nn
import numpy as np

# quite obvious the pattern here is y = 3x + 1
# Therefore out weight should = 3 and our bias should = 1
x = Tensor([[1, 2, 3, 4, 5]]).transpose()
y = Tensor([[4, 7, 10, 13, 16]]).transpose()

lin = nn.Linear(1, 1)

loss_fun = nn.SquaredLoss()

optimizer = nn.SGD(lin.parameters())


for i in range(1000):
    optimizer.zero_grad()

    out = lin(x)

    loss = loss_fun(out, y)

    loss.backward()

    optimizer.step()


w, b = lin.parameters()

print(w.value)
print(b.value)

# Prints 3, 1 just like it is expected to do.

# Also trains nearly instantly over 10k iterations










