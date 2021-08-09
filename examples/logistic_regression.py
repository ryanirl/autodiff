from autodiff.tensor import Tensor
import autodiff.nn as nn
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

### --- Hyperparameters --- ###

n = 100
eta = 0.01
itters = 5000

###############################



### --- Defining Data --- ###

X, y = make_blobs(n_samples = n, centers = 2, random_state = 2)

X = Tensor(X)                  # (100, 2)
y = Tensor(y[:, np.newaxis])   # (100, 1)

#############################



### --- Building the Model --- ###

class LogisticRegression:
    def __init__(self):

        self.layers = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.layers(X)

    def parameters(self):
        return self.layers.parameters()


##################################



### --- Training --- ###

model = LogisticRegression()

loss_fun = nn.BinaryCrossEntropy()

optimizer = nn.SGD(model.parameters(), lr = 0.01)

for i in range(5000):
    optimizer.zero_grad()

    out = model.forward(X)

    loss = loss_fun(out, y)

    if i % 50 == 0: print("Loss at step {} is: {}".format(i, np.sum(loss.value)))

    loss.backward()

    optimizer.step()

    # Zero grad doesn't include this (working on a fix)
    X.grad = 0
    y.grad = 0

########################



### --- Visualizing --- ###

arr = model.parameters()

weight0 = arr[0].value[0]
weight1 = arr[0].value[1]
bias = arr[1].value[0]

print("weights: [{}, {}] | Bias: [{}]".format(weight0, weight1, bias))

Y = y.value
X = X.value

for i in range(n):
    if (Y[i] == 1):
        plt.scatter(X[i][0], X[i][1], color="green") 
    else:
        plt.scatter(X[i][0], X[i][1], color="blue") 

    
x = np.linspace(-10, 10, 10)

hyperplane = (-(weight0 / weight1) * x) - (bias / weight1)

plt.plot(x, hyperplane, '-', color = "red")

plt.show()

###########################


