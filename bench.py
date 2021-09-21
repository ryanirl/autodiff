# This example was inspired by Andrej Karpathy's micrograd moon_demo which can
# be found here: https://github.com/karpathy/micrograd/blob/master/demo.ipynb
# His project was an initial inspiration for mine and I wanted to see how my
# implimentation would hold up against this.

from autodiff.tensor import Tensor
import autodiff.nn as nn

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

n = 100

X, y = make_moons(n_samples = n)

X = Tensor(X)
y = Tensor(y[:, np.newaxis])

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

optimizer = nn.Adam(model.parameters(), lr = 0.001)
loss_fun = nn.SigmoidBinaryCrossEntropy()

out = model.forward(X) 

loss = loss_fun(out, y)

loss.build_topo()

# Using updates rather than recomputaiton
# reduced time by 23% on this test.
# Though this may give up some functionality,
# I am unsure what kind of functionality I am
# giving up yet though.
start = time.time()

for i in range(50000):
    loss.topo_update()

    optimizer.step()

print(time.time() - start)


X = X.value
y = y.value


### --- Visualization --- ###

h = 0.01 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Xmesh = np.c_[xx.ravel(), yy.ravel()]

inputs = Tensor(Xmesh)

scores = model.forward(inputs)

Z = scores.value.reshape(xx.shape)

fig = plt.figure()

plt.contourf(xx, yy, Z, levels = 1, cmap = plt.cm.ocean, alpha = 0.9)

plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.ocean) 

plt.show()


            

