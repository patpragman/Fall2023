from statistics import stdev as sample_stdev
from statistics import pstdev as pop_stdev
import numpy as np
from numpy.linalg import inv, det

A = np.array([1, 2, 4, 3]).reshape((2, 2))
print(A)

print(5*inv(A))

A = np.array([1, 4, 7, 2, 5, 8, 3, 6, 10]).reshape((3,3))
print(A)

print(det(A))