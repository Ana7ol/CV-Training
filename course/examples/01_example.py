# The mathematical toolkit
import numpy as np
x = np.array([2., 3.])
w = np.array([1., -1.])
z = w @ x + 2.
p = 1 / (1 + np.exp(-z))
grad_w = 0.2 * x
print(z, p, grad_w)
