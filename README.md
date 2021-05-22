# Autodiff

<br />

<p align="center">
 <img src="./img/autodiff_img.png" width="90%">
</p>

-------

**AutoDiff is intended for educational purposes only.**

**WORK IN PROGRESS*

AutoDiff is a lightweight reverse-mode automatic differentiation (a.k.a
backpropagation) libary written in Python with Numpy vectorization.
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
done building the support for basic neural nets. (Why? Because I can.)

Now with support for user-defined primitive functions!

---

<!-- TODO -->
### TODO:
 - Add Examples and Tests (tests)
 - Neural Net functionality (nn.py)
 - Optimizations: Adam, Momentum, SGD, RMSProp (optim.py)
 - Convolutions (???)
 - DOCS & Draw.io

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
c = Tensor(4)

z = a + b
y = z ** 2
x = y * b
w = x * a
v = w ** 2
u = a * b
t = v * u
s = c * b
r = t - s

r.backward()

print("value: ({}) should be 134988".format(r.value))
print("grad wrt A: ({}) should be 310500".format(a.grad))
print("grad wrt B: ({}) should be 242996".format(b.grad))
print("grad wrt C: ({}) should be -3".format(c.grad))

```

---

You can even define your own primitive functions. An example of a user defined
primitive function may be:

```python

import numpy as np
from tensor import Tensor, OP
from lambdas import grad_fun, value_fun
from utils import primitive, check

value_fun["linear"] = (lambda w, x, b: ((w.value * x.value) + b.value))

# multiplying each gradient by "g" is requied by the chain rule
grad_fun["linear"] = (lambda g, w, x, b, z: (g * x.sum(), g * w.value, g * 1))

# Check(x, Tenor) just garentees that some x is not Tensor
@primitive(Tensor)
def linear(self, w, b):
    return OP("linear", check(w, Tensor), self, check(b, Tensor))

x = Tensor([1, 1, 3], requires_grad = True)
y = Tensor([1, 4, 10], requires_grad = False)
w = Tensor([1])
b = Tensor([2])

test = x.linear(w, b)
test.backward()

print(w.grad)
print(b.grad)
print(x.grad)

```



<!-- LISCENCE -->
### LISCENCE

#### MIT
