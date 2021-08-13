from autodiff.tensor import Tensor


class Module:
    def __init__(self):
        self.params = []

    def parameters(self):
        return self.params


class Sequential(Module):
    def __init__(self, *layers): 
        self.params = []
        self.layers = layers

        for layer in layers:
            self.params += layer.parameters()

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)

        return X
    
        
    def forward(self, X): return self.__call__(X)
