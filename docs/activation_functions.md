# Activation Functions Documentation

---

Table of Contents / TOOD:
 - Softmax
 - Sigmiod
 - ReLU
 - Leaky-ReLU
 - TanH

<br />

---


## Softmax:

*ROUGH DRAFT!*

**The Forward Pass:**

The forward pass of Softmax is the vanilla softmax but adjusted for numeric
stability. Becuase softmax has the property that Softmax(X - c) = Softmax(X)
for some constant c. We can stop Softmax from potentially giving us invalid 
numbers by subtracting the maximum value in out array from itself before 
exponentiating.


Softmax(X) = np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)), axis = 1, keepdims = True)


**The Backward Pass:**


Given ingrad and z = Softmax(X)


The derivation of the Softmax Activation Function derivative can be found online.
Most standard implementation that I have found today use the Kronecker Delta though
I have found a way to work around it. I will publish my notes on this eventually.

Long story short most people would do something like this NON-VECTORIZED APPROACH:

ingrad.dot(np.diag(z) - (z.T.dot(z)))

Though using some algebra you can rewrite this as:

(ingrad * z) - (z.T.dot(z)).dot(ingrad)

I haven't seen anyone purpose this solution. During testing it performed just about the same
speed, though it's simpler to understand and you don't have to compute the diag matrix w.r.t. each
row which is a pain to work out (maybe something like np.apply_along_axis(np.diag, 1, z)). And we
replace the diag(z) with a element wise multiplication of ingrad * z which is simple in nature.

The derivative of Softmax() w.r.t X is: 

 = (g * z) - (np.einsum('ijk,ik->ij', z[..., None] * z[:, None, :], g))])

where z[..., None] * z[:, None, :] is the vectorized dyadic matrix of each row in z.


Resources that helped me:

For the jacobian of Softmax this resource really helped:
https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy

The following articles helped me put together my vectorized version:
https://stackoverflow.com/questions/36279904/softmax-derivative-in-numpy-approaches-0-implementation
https://themaverickmeerkat.com/2019-10-23-Softmax/

For some intuitive understanding of softmax I recommend this lecture 
from Standfords CS321n Class: (Softmax around 37:00 in the video)
https://www.youtube.com/watch?v=h7iBpEHGVNc&t=3032s





<br />

---

## Sigmiod

NEED TO DO

**The Forward Pass:**
**The Backward Pass:**

<br />


---


## ReLU

NEED TO DO

**The Forward Pass:**
**The Backward Pass:**

<br />


---


## Leaky-ReLU

NEED TO DO

**The Forward Pass:**
**The Backward Pass:**

<br />


---


## TanH

NEED TO DO

**The Forward Pass:**
**The Backward Pass:**

<br />

---


Liscense: MIT

