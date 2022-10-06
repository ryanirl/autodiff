from autodiff.tensor import Tensor
import autodiff.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_iris

import numpy as np


### --- Preparing the Data --- ### 

X, y = load_iris(return_X_y = True)

label_binarizer = LabelBinarizer()

label_binarizer.fit(range(max(y) + 1))

y = label_binarizer.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)

X_train = Tensor(X_train, requires_grad = False)
y_train = Tensor(y_train, requires_grad = False)


### -- Training --- ###

loss_fun = nn.CrossEntropy()

weight = Tensor.uniform(4, 3)
bias = Tensor.uniform(1, 1)

params = [weight, bias]

optimizer = nn.Adam(params, lr = 0.001, eps = 1e-7)

for i in range(10000):
    optimizer.zero_grad()
    
    linear = X_train.dot(weight) + bias

    preds = linear.softmax()

    loss = loss_fun(preds, y_train) 

    if i % 50 == 0: print("Loss at step {} is: {}".format(i, np.sum(loss.value) / X_train.shape[0]))
    
    loss.backward() 

    optimizer.step()


### --- Checking the Testing Data --- ###

X_test, y_test = Tensor(X_test), Tensor(y_test)

# Making the prediction with our weights
linear = X_test.dot(weight) + bias
preds = linear.softmax()

# Setting the highest value to 1 and the rest to 0 in each row
preds = np.eye(preds.value.shape[1])[preds.value.argmax(1)]

correct = 0
incorrect = 0
for i in range(len(preds)):
    if (preds[i] == y_test.value[i]).all():
        correct += 1 
    else: 
        incorrect += 1


print()
print("Test Data: ")
print("---------------------------------------")
print("Amount predicted correctly:    | {} ".format(correct))
print("---------------------------------------")
print("Amount predicted in-correctly: | {}".format(incorrect))
print("---------------------------------------")
print()





