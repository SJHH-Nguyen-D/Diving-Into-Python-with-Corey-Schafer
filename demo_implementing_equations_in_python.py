import numpy as np 
import math
from operator import add
from scipy.stats import linregress

# implementing the factorial equation from scratch in python

def factorial(n):
	""" Factorial Function is n*n-1*n-2...n-n """
	res=1
	# starting from 1*1 to n. remember that python excludes the terminating number so we have to add 1 to n
	for i in range(1, n+1):
		res = res*i
	return res

print("{} is the factorial of 5!\n".format(factorial(5)))

# Plotting factorial vs number used for factorial
import matplotlib.pyplot as plt 

n = np.arange(1, 10, 1)
y = [factorial(i) for i in n]

fig, ax = plt.subplots()
ax.plot(n, y)

ax.set(xlabel='n', ylabel='n!',
       title='n vs n!')
ax.grid()

# fig.savefig("test.png")
# plt.show()

######################################################################

# implementing euler's number function 2.17
def e():
	""" Euler's number is defined as 
	the sum of 1/n! until infinity or a very large number
	"""
	res = 0
	for n in range(100):
		res = res + (1/factorial(n))
	return res

print(f"This is euler's number: {e()}\n")

######################################################################

def avg(X):
	""" The averaging function from scratch for a collection of numbers X """
	res = 0
	for x in X:
		res = res + x
	res = res * 1/len(X)
	return res

######################################################################

def average(X):
	""" The average function without the looping and indexing through a list"""
	res = 0
	for x in X:
		res = res + x
	res = res/len(X)
	return res

np.random.seed(0)
X = np.random.randint(low=1, high=100, size=10)

print("If {} == {}, print: {}\n".format(avg(X), average(X), avg(X) == average(X)))

######################################################################

# implement the variance functionsd
def var(X):
	""" 
	The variance function given a collection of numbers.
	The one trick to this is that there are two for loops 
	for summation in the variance function.
	"""
	res = 0
	for i in X:
		for j in X:
			res = res + 1.0/2.0 * (i-j)**2
	res = res * 1/(len(X)**2)
	return res

print("The variance for an array of numbers X is: {}\n".format(var(X)))

######################################################################
def variance(X):
	""" alternative implementation of the variance function """
	res = 0
	for x in X:
		res = res + (x - average(X))**2
	res = res/len(X)
	return res


def std_dev(X):
	""" standard deviation implementation dependent on the variance calculation """
	return variance(X)**(1.0/2.0)

dog_heights = [600,470,170,430,300]
print(
	"Mean of dog heights: {}\nVariance of dog heights: {}\nStandard Deviation of dog heights: {}\n".format(
	average(dog_heights),
	variance(dog_heights), 
	std_dev(dog_heights)
	)
)

######################################################################

def euclidean_distance(A, B):
	""" The euclidean distance between two collections of numbers """
	distance = 0
	for a, b in zip(A, B):
		distance = distance + (b-a)**2
	distance = distance**(1.0/2.0)
	return distance

A = np.random.normal(size=10)
B = np.random.normal(size=10)
print("The euclidean distance between arrays A and B is : {}\n".format(euclidean_distance(A=A, B=B)))

######################################################################

def sigmoid(y):
	""" Sigmoid function """
	return 1/(1 + e()**-y)

print("The sigmoid function result of 3 is {}\n".format(sigmoid(3)))

# plot the sigmoid function as a function of its value against X
a = np.arange(1, 20+1)
fig, ax = plt.subplots()
ax.plot(a, sigmoid(a))
ax.set(xlabel="x", ylabel="sigmoid value of x", title="x vs sigmoid")
ax.grid()
# plt.show()

######################################################################

def coeff(X, Y):
	""" Estimate the weigtht coefficients of each predictor variable """
	numerator = 0
	for x, y in zip(X, Y):
		numerator += x-average(X) * y-average(Y)

	denominator = 0
	for x in X:
		denominator += x - average(X)**2

	return numerator/denominator 


def intercept(X, Y, coeff):
	""" calculate the y-intercept of the linear regression function """
	return average(Y) - coeff * average(X)


def linear_regression_fit(X, Y, random_error=np.random.rand(1)):
	""" Linear Regression function """
	y_pred = np.empty(shape=len(Y))
	for x, y in zip(X, Y):
		np.append(y_pred, intercept(X, Y, coeff=coeff(X, X)) + coeff(X, Y)*x + random_error)
	
	return y_pred


np.random.seed(0)
X_train = np.random.randint(1, 10, size=10)
y_truth = np.random.randint(1, 10, size=10)

from pprint import pprint
y_pred = linear_regression_fit(X_train, y_truth)
print("{}".format(y_pred))

# # subplot 1
plt.subplot(1, 3, 1)
plt.plot(X, y_truth, 'o-')
plt.plot(linregress(X, y_truth))
plt.title('Truth graph')
plt.ylabel('Y_truth')

# # subplot 2
plt.subplot(1, 3, 2)
plt.plot(X, linear_regression_fit(X_train, y_truth), '.-')
plt.plot(linregress(X_train, y_truth))
plt.title('Predicted Graph')
plt.ylabel('Y_pred')

plt.legend()
plt.grid()
plt.show()

######################################################################

def logistic_error_function(X, Y):
	""" 
	logistic error function for Bernoulli/Binary outcome classification:
	"""
	j_theta = 0
	for x, y in zip(X, Y):
		j_theta = j_theta + y*math.log(sigmoid(x)) + (1-y)*math.log(1-sigmoid(x))
	j_theta = j_theta * (-1/len(X))
	return j_theta

######################################################################

def logistic_regression_fit(X, coeff, intercept):
	""" Logit Link Function """
	# sigmoid function
	prob_a = sigmoid(linear_regression_fit(X, coeff, intercept))
	return prob_a

######################################################################
