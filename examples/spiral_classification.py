# This example is froma case study done in Stanford's CS321n class. The
# information can be found here: https://cs231n.github.io/neural-networks-case-study/

from autodiff.tensor import Tensor
import autodiff.nn as nn

import matplotlib.pyplot as plt
import numpy as np


### --- Dataset as Defined in the Case Study --- ###

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes

X = np.zeros((N * K, D)) # data matrix (each row = single example)
y = np.zeros(N * K, dtype = 'uint8') # class labels

for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1 , N) # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2 # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j


### --- One-Hot Encoding as required by Autodiff's CCE --- ###

temp = np.zeros((y.size, y.max() + 1))
temp[np.arange(y.size), y] = 1

y = Tensor(temp, requires_grad = False)
X = Tensor(X, requires_grad = False)


### --- Building the Model --- ###

class Model:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Linear(2, 100),
#            nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(100, 3),
            nn.Softmax()
        )

    def parameters(self):
        return self.layers.parameters()

    def forward(self, X):
        return self.layers(X)


### --- Training --- ###

loss_fun = nn.CategoricalCrossEntropy()

model = Model()

optimizer = nn.Adam(model.parameters())

for i in range(1000):
    optimizer.zero_grad()

    out = model.forward(X)

    loss = loss_fun(out, y)

    if i % 50 == 0: print("loss at {} is: {}".format(i, np.sum(loss.value) / 300))

    loss.backward()

    optimizer.step()


### --- Visualization --- ###

X = X.value
y = y.value

h = 0.01 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Xmesh = np.c_[xx.ravel(), yy.ravel()]

inputs = Tensor(Xmesh)

scores = model.forward(inputs)

Z = scores.value

Z = np.argmax(Z, axis = 1)

Z = Z.reshape(xx.shape)

fig = plt.figure()

plt.contourf(xx, yy, Z, levels = 2, cmap = plt.cm.ocean, alpha = 0.9)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 15, cmap = plt.cm.ocean) 

plt.show()





