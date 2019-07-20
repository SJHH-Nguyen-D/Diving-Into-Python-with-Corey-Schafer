import numpy as np
import math
from operator import add
from scipy.stats import linregress


class DogGroup:
    def __init__(self, height, weight, breed):
        self.height = height
        self.weight = weight
        self.breed = breed


# implementing the factorial equation from scratch in python


def factorial(n):
    """ Factorial Function is n*n-1*n-2...n-n """
    res = 1
    # starting from 1*1 to n. remember that python excludes the terminating number so we have to add 1 to n
    for i in range(1, n + 1):
        res = res * i
    return res


print("{} is the factorial of 5!\n".format(factorial(5)))

# Plotting factorial vs number used for factorial
import matplotlib.pyplot as plt

n = np.arange(1, 10, 1)
y = [factorial(i) for i in n]

fig, ax = plt.subplots()
ax.plot(n, y)

ax.set(xlabel="n", ylabel="n!", title="n vs n!")
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
        res = res + (1 / factorial(n))
    return res


print(f"This is euler's number: {e()}\n")

######################################################################


def avg(X):
    """ The averaging function from scratch for a collection of numbers X """
    res = 0
    for x in X:
        res += x
    res = res * 1 / len(X)
    return res


######################################################################


def average(X):
    """ The average function without the looping and indexing through a list"""
    res = 0
    for x in X:
        res += x
    res = res / len(X)
    return res


np.random.seed(0)
X = np.random.randint(low=1, high=100, size=10)

print("If {} == {}, print: {}\n".format(avg(X), average(X), avg(X) == average(X)))

######################################################################

dog_heights_train = [600, 470, 170, 430, 300]
dog_spots = [10, 20, 5, 13, 18]
cat_heights = [500, 750, 120, 300, 123]

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
            res = res + 1.0 / 2.0 * (i - j) ** 2
    res = res * 1 / (len(X) ** 2)
    return res


print("The variance for an array of numbers X is: {}\n".format(var(X)))


def variance(X):
    """ alternative implementation of the variance function """
    res = 0
    for x in X:
        res = res + (x - average(X)) ** 2
    res = res / len(X)
    return res


def variance_1(values):
    return sum([(x - average(values)) ** 2 for x in values]) / len(values)


def std_dev(X):
    """ standard deviation implementation dependent on the variance calculation """
    return variance(X) ** (1.0 / 2.0)


print("### Dog heigh summary statistics ###\n")
print(
    "Mean of dog heights: {}\nVariance of dog heights: {}\nStandard Deviation of dog heights: {}\n".format(
        average(dog_heights_train),
        variance(dog_heights_train),
        std_dev(dog_heights_train),
    )
)
print("####################################\n")


def covariance(X, Y):
    """ Covariance is a generatlization of correlation. Correlation describes the relationship between 
    two groups of numbers, whereas covariance describes it between two or more groups of numbers
    return:
    sum((x(i) - mean(x)) * (y(i) - mean(y)))
    """
    res = 0
    for x, y in zip(X, Y):
        res += (x - average(X)) * (y - average(Y))
    return res


print(
    "This is the covariance between dog spots and dog heights: {}\n".format(
        covariance(dog_heights_train, dog_spots)
    )
)


######################################################################
############### Simple Linear Regression  ############################
######################################################################


def regression_regularizer(type="l1"):
    pass

def coeff(X, Y):
    """ Estimate the weigtht coefficients of each predictor variable """
    return covariance(X, Y) / variance(X)


def intercept(X, Y):
    """ calculate the y-intercept of the linear regression function """
    return average(Y) - coeff(X, Y) * average(X)


def simple_linear_regression(X_train, X_test, Y, random_error=np.random.random()):
    """ Simple Linear Regression function """
    y_pred = np.empty(shape=len(Y))

    b0, b1 = intercept(X_train, Y), coeff(X_train, Y)

    for x_test in X_test:
        np.append(y_pred, b0 + b1 * x_test + random_error)

    return y_pred


def root_mean_squared_error(actual, predicted):
    """ Loss function by which we use to evaluate our SLR model """
    sum_error = 0
    for act, pred in zip(actual, predicted):
        prediction_error = act - pred
        sum_error += prediction_error ** 2
    mean_error = sum_error / len(actual)
    return math.sqrt(mean_error)


# test simple linear regression
dog_weights_train = [4.5, 3.5, 11.1, 3.4, 2.3]

dog_heights_test = list(reversed(dog_heights_train))
dog_weights_test = list(reversed(dog_weights_train))

train_dataset = [list(i) for i in zip(dog_heights_train, dog_weights_train)]
test_dataset = [list(i) for i in zip(dog_heights_test, dog_weights_test)]

# fitting the SLR to get predictions
y_pred = simple_linear_regression(
    dog_heights_train,
    dog_heights_test,
    dog_weights_train,
    random_error=np.random.rand(1),
)

print(
    "This is the prediction of dog weights "
    "given new dog height information from the "
    "learned coefficients: {}\n".format(y_pred)
)

# evaluating the performance of the SLR
rmse = root_mean_squared_error(dog_weights_test, y_pred)
print("RMSE between the predicted and actual dog_weights is : {0:.3f}\n".format(rmse))


# Plotting the actual vs. the predicted values of the dog weights
fig, ax = plt.subplots()
ax.plot(dog_heights_test, dog_weights_test, "o-")
ax.plot(dog_heights_test, y_pred, ".-")
ax.plot(linregress(dog_heights_test, dog_weights_test))
ax.set(
    xlabel="Dog heights",
    ylabel="Dog weights",
    title="Dog heights vs dog weight predictions",
)
plt.grid()
# plt.show()

######################################################################


######################################################################
def sigmoid(y):
    """ Sigmoid function
    you can use math.e instead of e() if you choose you
    """
    return 1 / (1 + e() ** -y)


print("The sigmoid function result of 3 is {}\n".format(sigmoid(3)))

# plot the sigmoid function as a function of its value against X
a = np.arange(1, 20 + 1)
fig, ax = plt.subplots()
ax.plot(a, sigmoid(a))
ax.set(xlabel="x", ylabel="sigmoid value of x", title="x vs sigmoid")
ax.grid()
# plt.show()


def logistic_error_function(X, Y):
    """ 
	logistic error function for Bernoulli/Binary outcome classification:
	"""
    j_theta = 0
    for x, y in zip(X, Y):
        j_theta = (
            j_theta + y * math.log(sigmoid(x)) + (1 - y) * math.log(1 - sigmoid(x))
        )
    j_theta = j_theta * (-1 / len(X))
    return j_theta


def logistic_regression(X_train, X_test, Y):
    """ Logit Link Function """
    # sigmoid function
    prob_a = sigmoid(simple_linear_regression(X_train, X_test, Y))
    return prob_a


prob_a = logistic_regression(dog_heights_train, dog_heights_test, dog_weights_train)
print("This is the output of the logistic regression model:\n{}".format(prob_a))

fig, ax = plt.subplots()
ax.plot(dog_heights_test, dog_weights_test)
ax.plot(dog_heights_test, prob_a)
ax.plot(dog_heights_test, y_pred)
ax.set(
    xlabel="Dog heights (cm)",
    ylabel="Dog weights (kg)",
    title="Dog heights vs weights predictions",
)
plt.grid()
# plt.show()

