# Autodiff

<br />

<p align="center">
 <img src="./img/autodiff_img.png" width="90%">
</p>

-------

AutoDiff is a lightweight transparent reverse-mode automatic differentiation
(a.k.a backpropagation) library written in Python with NumPy vectorization.
AutoDiff works by breaking up larger user defined functions into primitive
operators (such as addition, muliplication, etc.) whos derivatives are
pre-defined. In the forward pass, Autodiff dynamically builds a computation
graph of the larger user defined function using these primitive operators as
nodes of the computation graph. Next, to calculate the respective derivatives
w.r.t.  each variable the chain rule is applied during the backwards pass of
the computation graph. Though there are various techniques for implementing
reverse-mode automatic differentiation. AutoDiff utilises Python's operator
overloading capabilities which is by far the simplier and more intuitive of the
methods. Autodiff is currently split into 2 levels:

**Level 1 (L1):** Introduces the base functionality of Autodiff. Level 1 defines the 
Tensor class, supports primitive operators, and includes a decorator that 
allows users to create custom "primitive" ops on top of Autodiff. 

**Level 2 (L2):** Everything inside of the NN folder. Level 2 adds on
top of Autodiff using the "register" decorator defined in Level 1. Level 2
introduces various loss functions, activation functions, and more.

**Dependencies:** NumPy.

---


Table of Contents
=================
   * [Supported Features](#currently-supported-nn-features)
   * [Installation](#installation)
   * [Usage](#usage)
        * Basic Usage
        * Building Simple MLP
        * User Defined Primitives
   * [Liscence](#liscence)

---


<!-- Currently Supported -->
## Currently Supported NN Features:

<details>
   <summary>Activation Functions:</summary>

</br>

| Activation Function  | Implimented |
| ----------- | -------------------- |
| ReLU        | :white_check_mark:   |
| Leaky ReLU  | :white_check_mark:   |
| PReLU       | :white_large_square: |
| Sigmoid     | :white_check_mark:   |
| Log Sigmoid | :white_large_square: |
| Softmax     | :white_check_mark:   |
| Log Softmax | :white_large_square: |
| TanH        | :white_check_mark:   |

</details>

<details>
   <summary>Loss Functions:</summary>

</br>

| Loss Function       | Implimented          |
| ------------------- | -------------------- |
| MAE (L1)            | :white_large_square: |
| MSE (L2)            | :white_check_mark:   |
| Binary CE           | :white_check_mark:   |
| Categorical CE      | :white_check_mark:   |
| Sigmoid & Soft w/CE | :white_check_mark:   |
| Hinge Loss          | :white_large_square: |

</details>

<details>
   <summary>Optimizers:</summary>

</br>

| Optimizer        | Implimented          |
| ---------------- | -------------------- |
| SGD              | :white_check_mark:   |
| SGD w/ Momentum  | :white_check_mark:   |
| Nestrov Momemtum | :white_check_mark:   |
| AdaGrad          | :white_check_mark:   |
| RMSProp          | :white_check_mark:   |
| Adam             | :white_check_mark:   |
| AdaDelta         | :white_large_square: |

</details>

<details>
   <summary>Layers:</summary>

</br>

| Layer         | Implimented          |
| ------------- | -------------------- |
| Linear        | :white_check_mark:   |
| Sequential    | :white_check_mark:   |
| 2DConvolution | :white_large_square: |
| Max Pooling   | :white_large_square: |
| Batch Norm    | :white_large_square: |
| Dropout       | :white_large_square: |

</details>


---

<!-- INSTALLATION -->
## Installation

For the moment, you have to clone the library to play with it.

---

<!-- USAGE -->
## Usage


<!-- BASIC USAGE -->
### Basic Example:

```python

from autodiff.tensor import Tensor

a = Tensor(2)
b = Tensor(3)

# This is the same as writing f(x) = ((a + b) * a)^2
# We just break it down into primitive ops.

z = a + b
y = z * a
x = y ** 2

# This is where the magic happens
x.backward()

print("value of x: ({})".format(x.value))
print("grad of x wrt a: ({})".format(a.grad))
print("grad of x wrt b: ({})".format(b.grad))

```

---

<!-- NN EXAMPLE -->
### NN Example:

Bulding an MLP to do multiclass Softmax classification is as simple as this:
This example in full detail can be found here: https://github.com/ryanirl/autodiff/blob/main/examples/spiral_classification.py

```python
from autodiff.tensor import Tensor
import autodiff.nn as nn

# Instantiating the Model
model = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 3),
    nn.Softmax()
)

# Defining the Loss
loss_fun = nn.CategoricalCrossEntropy()

# Defining the Optimizer
optimizer = nn.Adam(model.parameters())

# Training
for i in range(1000):
    optimizer.zero_grad()

    out = model.forward(X)

    loss = loss_fun(out, y)

    loss.backward()

    optimizer.step()

```

Plotting the decision boundry gives: 


<p align="center">
 <img src="./img/spiral_classification_img.png" width="90%">
</p>



---

<!-- USER DEFINED PRIMITIVES -->
### User Defined Primitives

You can even define your own primitive functions. An example of a user defined
primitive function may be: 

```python
from autodiff.tensor import Tensor, register
import numpy as np

# For simplicity
def e(x): return np.exp(x.value)

@register
class tanh:
    def forward(x):
        return (e(x) - e(-x)) / (e(x) + e(-x))

    def backward(g, x, z):
        return [g * (1.0 - (z.value ** 2))]


x = Tensor([1, 2, 3])
y = x.tanh()

y.backward()

print("The gradient of y wrt x: {}".format(x.grad))

# OUTPUTS: The gradient of y wrt x: [[0.41997434 0.07065082 0.00986604]]

```

---


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.




