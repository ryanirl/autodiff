from autodiff.tensor import Tensor

class ReLU:
    def __call__(self, X): return X.relu()

class Sigmoid:
    def __call__(self, X): return X.sigmoid()


### ----- NEED TO IMPLIMENT ----- ###

#class LeakyReLU:
#    def __call__(self, X): return X.leaky_relu()

#class Softmax:
#    def __call__(self, X): return X.softmax()

#class Tanh:
#    def __call__(self, X): return X.tanh()



# WOULD LINEAR BE AN ACTIVATION FUNCTION??
#class Linear:
#    def __call__(self, X): return X.Linear()











# MAYBE

#model = Sequential(
#    Linear(2, 2, activation = "tanh")
#)

