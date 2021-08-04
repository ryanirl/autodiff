# Autodiff

<br />

<p align="center">
 <img src="./img/autodiff_img.png" width="90%">
</p>

-------

**AutoDiff is intended for educational purposes only.**

**WORK IN PROGRESS*

AutoDiff is a lightweight reverse-mode automatic differentiation (a.k.a
backpropagation) library written in Python with Numpy vectorization.
AutoDiff works by breaking up larger user defined functions into primitive
operators (such as addition, muliplication, etc.) whos derivatives are
pre-defined. Autodiff then dynamically builds a computation graph of the larger
function using these primitive operators during a forward pass and then
applies the chain rule through the backwards pass of the computation
graph to compute the derivative of the larger fucntion wrt some variable.

Though there are various methods to do reverse-mode automatic differentiation.
AutoDiff works via Python's Operator Overloading abilities which is by far the
simplier and more intuitive of the methods. 

This project is still very much a work in progress and I plan on building this
into a mini deep learning library with support for training basic MLP's, 
CNN's, and more. I also plan on supporting higher-order derivatives once I'm 
done building the support for basic neural nets. 

Now with support for user-defined primitive functions!

---

<!-- TODO -->
### TODO:
 - Add Examples and Tests (tests)
 - Neural Net functionality (nn.py)
 - Optimizations: Adam, Momentum, SGD, RMSProp (optim.py)
 - Convolutions (???)

#### Might consider adding in the future:
 - Hardware Acceleration 

---

<!-- INSTALLATION -->
### Installation

At the moment just gonna have to clone the repo and make sure you have numpy
installed which is it's only dependency.

Not tested on Python2.

---

<!-- USAGE -->
### USAGE 

Basic Example Usage:

```python

from autodiff.tensor import Tensor

a = Tensor(2)
b = Tensor(3)

# This is the same as writing f(x) = ((a + b) * a)^2
# We just break it down into primitive ops

z = a + b
y = z * a
x = y ** 2

# This is where the magic happens
x.backward()

print("value: ({})".format(x.value))
print("grad wrt A: ({})".format(a.grad))
print("grad wrt B: ({})".format(b.grad))

```

---

I plan on simplifying this but right now you can even define your own primitive
functions. An example of a user defined primitive function may be: 

**Note #1:** To better understand this code and how works and how to add your own
primitive operators, I will be working on a guide soon. 

```python

import numpy as np
from tensor import Tensor, OP
from lambdas import grad_fun, value_fun
from utils import primitive, check

def e(x): return np.exp(x.value)

value_fun["tanh"] = (lambda x: (e(x) - e(-x)) / (e(x) + e(-x)))

# multiplying each gradient by "g" is requied by the chain rule
grad_fun["tanh"] = (lambda g, x, z: [(g * (1.0 - (z ** 2)))])


@primitive(Tensor)
def tanh(self):
    return OP("tanh", self);

x = Tensor([1, 2, 3])
y = x.tanh()

y.backward()

print("The gradient of y wrt x: {}".format(x.grad))

```



<!-- LISCENCE -->
### LISCENCE

#### MIT
