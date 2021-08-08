from autodiff.tensor import Tensor

# These classes allow you to work with them like they are
# layers rarther than simple tensor ops

# It allows us to do things like this:

#model = Sequential(
#    Linear(2, 2),
#    Relu(),
#    Linear(2, 1),
#)

class ReLU:
    def __call__(self, X): return X.relu()

class Sigmoid:
    def __call__(self, X): return X.sigmoid()

class Softmax:
    def __call__(self, X): return X.softmax()

class LeakyReLU:
    def __call__(self, X): return X.leaky_relu()

class Tanh:
    def __call__(self, X): return X.tanh()




