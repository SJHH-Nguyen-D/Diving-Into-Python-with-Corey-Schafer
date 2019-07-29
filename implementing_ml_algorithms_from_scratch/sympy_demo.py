from sympy import *
from sympy import sin, sqrt
from sympy.abc import x, n
from sympy.integrals import Integral

x = Symbol('x')

# example 1: finding the indefinite integral for function: x**2+8 = x**4 + 7*x**(3 + 8)
# reminder: Indefinite integral is an integral with no lower or upper limit specified.
integralex = Integral((x**2)+8,x)
print(integralex.doit()) # x**3/3 + 8*x

# example 2: integrating the same function above (x**2+8), but we will perform a definite integral with 
# respect to a lower limit of 2 and an upper limit of 4.
x= Symbol('x')

integralex= Integral((x**2)+8,(x,2,4)) # the second argument reads as: wrt x variable, with lower limit 2, and upper limit 4
print(integralex.doit())
