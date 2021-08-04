import numpy as np
from collections import defaultdict

grad_fun = defaultdict(lambda: "ERROR")
value_fun = defaultdict(lambda: "ERROR")

##### --- GRAD DEFINITIONS --- #####

# Binary
grad_fun["add"] = (lambda g, x, y, z: (g, g))
grad_fun["sub"] = (lambda g, x, y, z: (g, -g))
grad_fun["mul"] = (lambda g, x, y, z: (g * y.value, g * x.value))
grad_fun["pow"] = (lambda g, x, y, z: (g * np.power(y.value * x.value, y.value - 1), 0))
grad_fun["div"] = (lambda g, x, y, z: (g / y.value, g / (y.value ** 2)))


# Unary
# Haven't extensivly tested these yet so be cautious.
grad_fun["sigmoid"] = (lambda g, x, z: [g * (z * (1.0 - z))])
grad_fun["relu"] = (lambda g, x, z: [g * (x.value > 0)]) 
grad_fun["log"] = (lambda g, x, z: [g / x.value])



##### --- VALUE DEFINITIONS --- #####

# Binary
value_fun["add"] = (lambda x, y: x.value + y.value)
value_fun["sub"] = (lambda x, y: x.value - y.value)
value_fun["mul"] = (lambda x, y: x.value * y.value)
value_fun["div"] = (lambda x, y: x.value / y.value)
value_fun["pow"] = (lambda x, y: x.value ** y.value)

# Unary
value_fun["relu"] = (lambda x: np.maximum(x.value, 0)) # NEED TO TEST
value_fun["sigmoid"] = (lambda x: 1.0 / (1.0 + np.exp(-x.value))) # NEED TO TEST
value_fun["log"] = (lambda x: np.log(x.value))

### ---  END PRIM  --- ###



### --- STILL NEED TO TEST SO PROBABLY DON'T USE --- ###
grad_fun["dot"] = (lambda g, x, y, z: (np.dot(g, y.value.T), np.dot(x.value.T, g)))
value_fun["dot"] = (lambda x, y: np.dot(x.value, y.value))

grad_fun["exp"] = (lambda g, x, z: [(g * z)])
value_fun["exp"] = (lambda x: (np.exp(x.value)))







