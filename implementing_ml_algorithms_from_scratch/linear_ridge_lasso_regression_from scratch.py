from sklearn.datasets import *
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np 
import math

def lasso_regression_penalty(l1_lambda, feature_weights):
    """ l2_lambda is between 0 and positive infinity """
    try:
        lasso_penalty = l1_lambda * sum(map(lambda x: abs(x), feature_weights))
    except ValueError:
        raise ValueError("Lambda should be a value between 0 and positive infinity")
    return lasso_penalty


def ridge_regression_penalty(l2_lambda, feature_weights):
    """ l2_lambda is between 0 and positive infinity """
    try:
        ridge_penalty = l2_lambda * sum(map(lambda x: x**2, feature_weights))
    except ValueError:
        raise ValueError("Lambda should be a value between 0 and positive infinity")
    return ridge_penalty


def elasticnet_regression_penalty(lambda1, lambda2, feature_weights):
    """ elasticnet regression is just the addition of lasso and ridge regressions 
    :params lambda1: regularization parameter for LASSO regression penalty
    :params lambda2: regularization parameter for Ridge regression penalty
    """
    return lasso_regression_penalty(lambda1, feature_weights) + ridge_regression_penalty(lambda2, feature_weights)


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


def simple_linear_regression(X_train, X_test, y_train, y_test, random_error=np.random.random(), regularizer=None):
    """ Simple Linear Regression function is a univariate regressor"""
    y_pred = np.empty(shape=len(y_test))

    b0, b1 = intercept(X_train, y_train), coeff(X_train, y_train)

    for x_test in X_test:
        np.append(y_pred, b0 + b1 * x_test + random_error)

    return y_pred


def root_mean_squared_error(actual, predicted, regularizer=None):
    """ Loss function by which we use to evaluate our SLR model """
    
    sum_error = 0
    for act, pred in zip(actual, predicted):
        prediction_error = act - pred
        sum_error += prediction_error ** 2
    mean_error = sum_error / len(actual)
    error = math.sqrt(mean_error)

    # regularization... find somewhere to plug in the lambda values into the function depending on the regularization
    if regularizer == "l1":
        error = error + lasso_regression_penalty(lambda1, b1)

    elif regularizer == "l2":
       error = error + ridge_regression_penalty(lambda2, b1)

    elif regularizer == "elasticnet":
       error = error + elasticnet_regression_penalty(lambda1, lambda2, b1)

    return error


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
iris = load_iris()
X = iris.data 
y = iris.target
feature_1_x = X[:, 0]
print(feature_1_x)
mse_ridge = lasso_regression_penalty(0.2, feature_1_x)
print("The MSE after the ridge regression is: {}".format(mse_ridge))


# train_dataset = [list(i) for i in zip(dog_heights_train, dog_weights_train)]
# test_dataset = [list(i) for i in zip(dog_heights_test, dog_weights_test)]

# # fitting the SLR to get predictions
# y_pred = simple_linear_regression(
#     dog_heights_train,
#     dog_heights_test,
#     dog_weights_train,
#     dog_weights_test,
#     random_error=np.random.rand(1),
# )

# print(
#     "This is the prediction of dog weights "
#     "given new dog height information from the "
#     "learned coefficients: {}\n".format(y_pred)
# )

# # evaluating the performance of the SLR
# rmse = root_mean_squared_error(dog_weights_test, y_pred)
# print("RMSE between the predicted and actual dog_weights is : {0:.3f}\n".format(rmse))
