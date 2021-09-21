# This example was inspired by Andrej Karpathy's micrograd moon_demo which can
# be found here: https://github.com/karpathy/micrograd/blob/master/demo.ipynb
# His project was an initial inspiration for mine and I wanted to see how my
# implimentation would hold up against this.

# The point of bench.py is to benchmark my autodiff with a from scratch
# implimentation that doesn't have any of the overhead that automatic 
# differentiation has. See my from scratch implimentation here: 
# https://github.com/ryanirl/ml-basics/blob/main/deep_learning/moons_from_scratch.py
# 
# Current Speed:
# From Scratch on 50k epochs: 6.3 seconds avg
# Autodiff on 50k epochs: 10.5 seconds avg
#
# Without optimization I was averaging 18.9 seconds. Optimizing control flow,
# update rules, swtiching from Adam to SGD (because my from scratch is SGD),
# and the unbroadcast function has allowed me to cut this time down to 10.5s
# on average.
# 
# Everything is run on a 2019 Macbook Pro CPU. Ultimitelly this tells me that
# my autodiff implimentation has 4.2 seconds of overhead in this case that I 
# need to lower as much as I can. 
#
# Ideas for Optimization:
#    - Optimize optim classes. 
#    - Anything that has to compute at each epoch, that has a constant state
#      throughout the training needs to be optimized (IG: unbroadcast axis which
#      I just optimized)
#    - Exploiting any parellelism in the underlying AD Computation Graph structure.
#      This is very hard to exploit as almost everything relies on the previous
#      computation but I will keep my eye out.
# 
# -----------------------------------------------------------
# Fasted lucky run so far: 10.09 (so close to sub 10 seconds)
# -----------------------------------------------------------
# 


from autodiff.tensor import Tensor
import autodiff.nn as nn

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import time

N_SAMPLES = 100
LR = 0.001
EPOCHS = 50000

X, y = make_moons(n_samples = N_SAMPLES)
X = Tensor(X)
y = Tensor(y[:, np.newaxis])

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

optimizer = nn.SGD(model.parameters(), lr = LR)

loss_fun = nn.SigmoidBinaryCrossEntropy()

out = model.forward(X) 

loss = loss_fun(out, y)

loss.build_topo()

start = time.time()

for i in range(EPOCHS):
    loss.topo_update()

    optimizer.step()

print("{} EPOCHS Computed in: {}".format(EPOCHS, time.time() - start))

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


            

