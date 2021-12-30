# Autodiff

<br />

<p align="center">
 <img src="./img/autodiff_img.png" width="90%">
</p>

-------


*WORK IN PROGRESS*

AutoDiff is a lightweight reverse-mode automatic differentiation (a.k.a
backpropagation) library written in Python with Numpy vectorization.
AutoDiff works by breaking up larger user defined functions into primitive
operators (such as addition, muliplication, etc.) whos derivatives are
pre-defined. Autodiff then dynamically builds a computation graph of the larger
function using these primitive operators as nodes during a forward pass and then
applies the chain rule through a backwards pass of the computation graph to
compute the derivative of the larger fucntion wrt some variable. 

Though there are various methods to impliment reverse-mode automatic differentiation.
AutoDiff works via Python's Operator Overloading abilities which is by far the
simplier and more intuitive of the methods. 

Autodiff currently works in 2 levels. Level 1 is largely complete, minus some small
things I will inevitably end up moving around in the future. Level 2 is more "Deep 
Learning", is very messy, and is basically a rough draft.

**Level 1:** is the base functionality of Autodiff, this level defined the
Tensor class, supports primitive operators, and includes a decorator that 
allows others to create custom "primitive" ops on top of Autodiff.

**Level 2:** is essentially everything inside of the NN folder. Level 2 adds on
top of Autodiff by using the "register" decorator in Level 1 Autodiff to define 
additional "primitive" (would it be primitive?) operators such as certain loss 
functions, activation functions, and more. Level 2 is very much so a work in 
progress.

---


Table of Contents
=================
   * [Supported Features](#currently-supported-nn-features)
   * [TODO](#todo)
   * [Installation](#installation)
   * [Usage](#usage)
        * Basic Usage
        * Building Simple MLP
        * User Defined Primitives
   * [Inspiration](#inpiration)
   * [Liscence](#liscence)


---


<!-- Currently Supported -->
## Currently Supported NN Features:

| Activation Functions | Implimented |   | Loss Functions      | Implimented          |   | Optimizers       | Implimented          |
| ----------- | -------------------- | - | ------------------- | -------------------- | - | ---------------- | -------------------- |
| ReLU        | :white_check_mark:   |   | MAE (L1)            | :white_large_square: |   | SGD              | :white_check_mark:   |
| Leaky ReLU  | :white_check_mark:   |   | MSE (L2)            | :white_check_mark:   |   | SGD w/ Momentum  | :white_check_mark:   |
| PReLU       | :white_large_square: |   | Binary CE           | :white_check_mark:   |   | Nestrov Momemtum | :white_check_mark:   |
| Sigmoid     | :white_check_mark:   |   | Categorical CE      | :white_check_mark:   |   | AdaGrad          | :white_check_mark:   |
| Log Sigmoid | :white_large_square: |   | Sigmoid & Soft w/CE | :white_check_mark:   |   | RMSProp          | :white_check_mark:   |
| Softmax     | :white_check_mark:   |   | Hinge Loss          | :white_large_square: |   | Adam             | :white_check_mark:   |
| Log Softmax | :white_large_square: |   |                     |                      |   | AdaDelta         | :white_large_square: |
| TanH        | :white_check_mark:   |   |                     |                      |   |                  |                      |



| Layers        | Implimented          |   | Autodiff Functionality | Implimented          |
| ------------- | -------------------- | - | ---------------------- | -------------------- |
| Linear        | :white_check_mark:   |   | Higher Order Gradients | :white_large_square: |
| Sequential    | :white_check_mark:   |   | Hardware Acceleration  | :white_large_square: |
| 2DConvolution | :white_large_square: |   | Tape Based Methods     | :white_large_square: |
| Max Pooling   | :white_large_square: |   | Float32 Tensors        | :white_large_square: |
| Batch Norm    | :white_large_square: |   | Grad Check             | :white_large_square: |
| Dropout       | :white_large_square: |   |                        |                      |

---


<!-- TODO -->
## TODO:
 - Primitive Functions: aMax
 - Loss Functions: Hinge, MAE
 - Layers: Pooling, Batch Normalization, Dropout, etc.
 - Utils: One-Hot for CCE-Loss, Split for Test/Train (Tensors), Pre-Defined Datasets
 - Grad Check
 - Hardware Acceleration?
 - Lots of Examples
 - Lots of Documentation


---

<!-- INSTALLATION -->
## Installation

At the moment just gonna have to clone the repo and make sure you have numpy
installed (which is it's only dependency) if you want to play with it.

Not tested on Python2.

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

    X.grad = 0 # Required if X is Tensor
    y.grad = 0 # Required if y is Tensor

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


<!-- Inspiration -->
## Inspiration:

I initially built Autodiff to help me understand backpropagation and what goes
into making a large library such as Tensorflow or PyTorch. Now that I have a
solid understanding of backpropagation and modern machine learning libraries I
continue to add on to this project to expand my knowledge in how certain ideas
in Machine / Deep Learning work at the core. Today, I could pip install
detectron2 and transfer learn a Mask R-CNN using a pretrained ResNet50 backbone 
on the COCO Dataset to a specialized task of detecting ornements on a Christmas 
tree (writting this on December 26th) using about 15 lines of code and have a well 
performing SOTA instance segmentation model. Though however convinient that
level of abstraction may be for certain tasks that you want to whip up in an
hour or two, it's also dangerous. It's dangerous because as developers if we
are entrusted with training one of these models but you don't necessarily
understand the underlying technology of the model your training then your bound 
to make ineffecient decisions during the training and pre/post processing stages.
I myself am guily of training models I don't necessarily understand and so I keep 
Autodiff as a reminder to not give into that convenience without first understanding 
the underlying technology.



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.




