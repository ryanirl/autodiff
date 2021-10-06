# This example was inspired by Andrej Karpathy's micrograd moon_demo which can
# be found here: https://github.com/karpathy/micrograd/blob/master/demo.ipynb
# His project was an initial inspiration for mine and I wanted to see how my
# implimentation would hold up against this.

# The point of bench.py is to benchmark my autodiff with a from scratch
# implimentation that doesn't have any of the overhead that automatic 
# differentiation has. See my from scratch implimentation here: 
# https://github.com/ryanirl/ml-basics/blob/main/deep_learning/moons_from_scratch.py
# 
# Benchmark on 2019 Macbook Pro CPU
# ------------------------------------------------------------------------------
# | Scrach Implimentation | 50k Epochs |  6.3 seconds | Epoch Speed: 0.000126s |
# ------------------------------------------------------------------------------
# | Autodiff              | 50k Epochs | 10.5 seconds | Epoch Speed: 0.000210s |
# ------------------------------------------------------------------------------
#
# Without optimization I was averaging 18.9 seconds. Optimizing control flow,
# update rules, swtiching from Adam to SGD (because my from scratch is SGD),
# and the unbroadcast function has allowed me to cut this time down to 10.5s
# on average. This tells me that Autodiff has 4.2 seconds of overhead for
# this specific example. I am trying to reduce this as much as I can without
# sacrificing functionality.
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
# ------------------------------------------------------------
# Fasted lucky run so far: 9.171808004379272 | SUB 10 SECONDS!
# ------------------------------------------------------------
# 
# Oct 5th, FOUND THE BOTTLE NECK:
# Removing the "unbroadcast" from the addition operator (because this model only uses
# addition) reduces the total runtime to: 6.444812059402466 seconds. That's nearly a
# 30% increase in speed... YIKES
# 
# To expand on the idea started above. Just setting bias = False in the linear layers
# doubles the performance of this project.
# 
# Speed without Bias: 5.2552649974823 seconds.
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
    nn.Linear(2, 16, bias = False),
    nn.ReLU(),
    nn.Linear(16, 16, bias = False),
    nn.ReLU(),
    nn.Linear(16, 1, bias = False)
)

# With bias is 2x slower.
#model = nn.Sequential(
#    nn.Linear(2, 16),
#    nn.ReLU(),
#    nn.Linear(16, 16),
#    nn.ReLU(),
#    nn.Linear(16, 1)
#)

optimizer = nn.SGD(model.parameters(), lr = LR)

loss_fun = nn.SigmoidBinaryCrossEntropy()

out = model.forward(X) 

loss = loss_fun(out, y)

loss.toposort()


#import cProfile
#
#with cProfile.Profile() as pr:
start = time.time()
for i in range(EPOCHS):
    loss.topo_update()
    optimizer.step()

print("{} EPOCHS Computed in: {}".format(EPOCHS, time.time() - start))

#pr.print_stats()




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

#plt.show()


            

