import numpy as np
import scipy.stats as st


def newtons_method(x_0, derivative, f):
    return x_0 - f(x_0)/derivative(x_0)

f = lambda x: x**3 + 2*x - 1
df = lambda x:  3*x**2 + 2

x_0 = 1.5

x_1 = newtons_method(x_0, df, f)
print(x_1)
x_2 = newtons_method(x_1, df, f)
print(x_2)
x_3 = newtons_method(x_2, df, f)
print(x_3)