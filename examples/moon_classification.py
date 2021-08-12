# This example was inspired by Andrej Karpathy's micrograd moon_demo which can
# be found here: https://github.com/karpathy/micrograd/blob/master/demo.ipynb
# His project was an initial inspiration for mine and I wanted to see how my
# implimentation would hold up against this.

from autodiff.tensor import Tensor
import autodiff.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

n = 100

X, y = make_moons(n_samples = n)

X = Tensor(X)
y = Tensor(y[:, np.newaxis])

class Model:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, X):
        return self.layers(X)

    def parameters(self):
        return self.layers.parameters()


model = Model()

loss_fun = nn.SigmoidBinaryCrossEntropy()

# This thing converges instantly with Adam :)
optimizer = nn.Adam(model.parameters())

for i in range(500):
    optimizer.zero_grad()

    out = model.forward(X)

    loss = loss_fun(out, y)

    loss.backward()

    if i % 100 == 0: print("loss after step {} is: {}".format(i, loss.value))

    optimizer.step()

    X.grad = 0
    y.grad = 0


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


            

