#*Dimensions: X = (150, 4) ; Weights = (4,3) ; y = (150, 3)*

from autodiff.tensor import Tensor
import autodiff.nn as nn
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

### --- Preparing the Data --- ### 

X, y = load_iris(return_X_y = True)

label_binarizer = LabelBinarizer()

label_binarizer.fit(range(max(y) + 1))

y = label_binarizer.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)

X_train, y_train = Tensor(X_train), Tensor(y_train)

################################## 



### -- Training --- ###

loss_fun = nn.CrossEntropy()

weight = Tensor.uniform(4, 3)
bias = Tensor.uniform(1, 1)

params = [weight, bias, X_train, y_train]

for i in range(10000):
    linear = X_train.dot(weight) + bias

    preds = linear.softmax()

    loss = loss_fun(preds, y_train) 

    if i % 50 == 0: print("Loss at step {} is: {}".format(i, np.sum(loss.value) / X_train.shape[0]))
    
    loss.backward() 

    # The gradients were exploding so I clipped them using this technique 
    # This might be a bug with my backprop so I will be looking into this
    weight.value = weight.value - 0.01 * (weight.grad / np.linalg.norm(weight.grad))
    bias.value = bias.value - 0.0001 * (bias.grad)

    # Setting the grads to 0
    for param in params:
        param.grad = 0


######################



### --- Checking the Testing Data --- ###

X_test, y_test = Tensor(X_test), Tensor(y_test)

# Making the prediction with our weights
linear = X_test.dot(weight) + bias
preds = linear.softmax()

# Setting the highest value to 1 and the rest to 0 in each row
preds = np.eye(preds.value.shape[1])[preds.value.argmax(1)]

print(preds)

print(y_test.value)

correct = 0
incorrect = 0
for i in range(len(preds)):
    if (preds[i] == y_test.value[i]).all():
        correct += 1 
    else: incorrect += 1

# When testing on my own the worst I got 35 correct and 3 incorrect.
# The best being 100% corrrect 
print("Amount predicted correctly: {}".format(correct))
print("Amount predicted in-correctly: {}".format(incorrect))

#########################################





