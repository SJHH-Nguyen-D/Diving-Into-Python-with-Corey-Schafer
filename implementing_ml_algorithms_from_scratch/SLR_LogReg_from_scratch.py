######################################################################
############### Simple Linear Regression ######################
######################################################################
######################################################################

import math
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def average(X):
    """ The average function without the looping and indexing through a list"""
    res = 0
    for x in X:
        res += x
    res = res / len(X)
    return res

def variance(values):
    return sum([(x - average(values)) ** 2 for x in values]) / len(values)

def std_dev(X):
    """ standard deviation implementation dependent on the variance calculation """
    return math.sqrt(variance(X))

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


def coeff(X, Y):
    """ Estimate the weigtht coefficients of each predictor variable """
    return covariance(X, Y) / variance(X)


def intercept(X, Y):
    """ calculate the y-intercept of the linear regression function """
    return average(Y) - coeff(X, Y) * average(X)


def simple_linear_regression(X_train, X_test, y_train, y_test, random_error=np.random.random()):
    """ Simple Linear Regression function """
    y_pred = np.empty(shape=len(y_test))

    b0, b1 = intercept(X_train, y_train), coeff(X_train, y_train)

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


def evaluate_SLR(dataset, algorithm):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(dataset, test_set)
    print(predicted)
    actual = [row[-1] for row in dataset]
    rmse = root_mean_squared_error(actual, predicted)
    return rmse


# test simple linear regression
dog_heights_train = [600, 470, 170, 430, 300]
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
    dog_weights_test,
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
# fig, ax = plt.subplots()
# ax.plot(dog_heights_test, dog_weights_test, "o-")
# ax.plot(dog_heights_test, y_pred, ".-")
# ax.plot(linregress(dog_heights_test, dog_weights_test))
# ax.set(
#     xlabel="Dog heights",
#     ylabel="Dog weights",
#     title="Dog heights vs dog weight predictions",
# )
# plt.grid()
# plt.show()


# Simple Linear Regression with a UCR dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Shape of X_train: {}\nShape of X_test: {}\n".format(X_train.shape, X_test.shape))
print("Shape of y_train: {}\nShape of y_test: {}\n".format(y_train.shape, y_test.shape))

# THIS BREAKS THE COMPUTER
# y_pred = simple_linear_regression(X_train, X_test, y_train, y_test, random_error=np.random.rand(1))

# # evaluating the performance of the SLR
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE between the predicted and actual diabetes ratings is : {0:.3f}\n".format(rmse))

# # Plotting the actual vs. the predicted values of the dog weights
# fig, ax = plt.subplots()
# ax.plot(X_test, y_test, "o-")
# ax.plot(X_test, y_pred, ".-")
# ax.plot(linregress(X_test, y_test))
# ax.set(
#     xlabel="Dog heights",
#     ylabel="Dog weights",
#     title="Dog heights vs dog weight predictions",
# )
# plt.grid()
# plt.show()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Sklearn's Implementation of Linear Regression #
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
from sklearn.linear_model import LinearRegression 
from pprint import pprint

lm = LinearRegression().fit(X_train, y_train)
y_pred = lm.predict(X_test)

df_slr_results = pd.DataFrame({"Actual Value": y_test.flatten(), "Predicted Value": y_pred})
print(df_slr_results.head())

print("SKLEARN's linear model scoring on the diabetes dataset: {}\n".format(lm.score(X_test, y_test)))
pprint("These are the weights from the linear model\n{}\n".format(lm.coef_))

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
diabetes_df = pd.DataFrame(diabete)
diabetes_df.head()

######################################################################
############### Logistic Regression Classifier ######################
######################################################################
######################################################################

def factorial(n):
	res = 1
	for i in range(1, n+1):
		res = res + i
	return res

def e():
    """ Euler's number is defined as 
	the sum of 1/n! until infinity or a very large number
	"""
    res = 0
    for n in range(100):
        res = res + (1 / factorial(n))
    return res

print("This is euler's number: {}".format(e()))


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


def logistic_regression(X_train, X_test, Y_train, Y_test):
    """ Logit Link Function """
    # sigmoid function
    prob_a = sigmoid(simple_linear_regression(X_train, X_test, Y_train, Y_test))
    return prob_a


prob_a = logistic_regression(dog_heights_train, dog_heights_test, dog_weights_train, dog_weights_test)
print("This is the output of the logistic regression model:\n{}".format(prob_a))

# fig, ax = plt.subplots()
# ax.plot(dog_heights_test, dog_weights_test)
# ax.plot(dog_heights_test, prob_a)
# ax.plot(dog_heights_test, y_pred)
# ax.set(
#     xlabel="Dog heights (cm)",
#     ylabel="Dog weights (kg)",
#     title="Dog heights vs weights predictions",
# )
# plt.grid()
# plt.show()


# Logistic Regression Classification with UCR dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.target_names

