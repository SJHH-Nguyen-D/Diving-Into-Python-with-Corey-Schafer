import numpy as np 
import pandas as pd 
from sklearn.datasets import *

iris = load_iris()
X = iris.data 
y = iris.target 
iris = pd.DataFrame(
	np.c_[iris["data"], iris["target"]],
	columns=iris['feature_names']+['target']
)

beeb = np.c_[X, y]
# print(beeb)

# def get_item(values):
# 	return values[4]

# iris_feature_columns = list(iris.columns[iris.columns !="target"])
# a = iris_feature_columns + ["adams", "orphan"]
# b = sorted(a, key=get_item, reverse=True)
# print(b)

# List operations practice
a = [1, 2, 3, 4, 5]
print(sum(a))
b = sorted(a, reverse=True)

print(a)
print(b)

# dot product of two lists
c = np.dot(a, b)
print(c)


# element wise multiplication of two lists
e = lambda x, y: x * y
d = [e(x, y) for x, y in zip(a, b)]
print(d)

# appending a list to the end of another list
# similar to unioning
f = a+b
print(f)

# element wise, pairing of two lists, where each pair is a list
# similar to merging
g = np.c_[a, b]
print(g)

# conditional filtering with lists
h = [a for a in b if a%2==0]
print(h)


############# simple map function ##############
i = list(map(lambda x: x**2, a))
print("Output of the first map function: {}".format(i))

# using list of map functions
add = lambda x: x+x
multiply = lambda x: x*x
funcs = [multiply, add]
for i in range(5):
	value = list(map(lambda x: x(i), funcs))
	print(value)

"""
since funcs is a list of two functions, the result of print would be the result of those
two functions being applied to i, which is the numeric index number of the range function
"""

######## using filtering function ##########

number_list = range(-5, 5)
less_than_two = list(filter(lambda x: x<2, number_list))
print("This is the result of the filter operation: {}".format(less_than_two))
"""
filter ressesmbels a for-loop but it is a builtin function and faster

within the number_list list, we apply a lambda condition to the filter operation
which inspects each element of the number_list and checks those that pass the condition
those elements that pass the filtering condition are put into the outer list function
"""

####### using reduce function ############

# we first compare using the reduce operation with how
# you would apply it using the vanilla method

product = 1
p = range(-10, 10)
for num in p:
	product = product * num 
print("This is the result of the vanilla 'filter' operation: {}".format(product)) 

# we now use the built-in reduce function from the built-in functools module in python
from functools import reduce 

# note that the map function inside the reduce aggregation method 
# being fed into the reduce function is enclosed in parentheses
product = reduce((lambda x, y: x * y), p)
print("This is the result of using" \
	" the built-in reduce function: {}.\nNote"\
	" that this solution is much more elegant "\
	"than the first method".format(product))

