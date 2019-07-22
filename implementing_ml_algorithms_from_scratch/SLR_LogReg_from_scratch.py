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


def simple_linear_regression(
    X_train, X_test, y_train, y_test, random_error=np.random.random()
):
    """ Simple Linear Regression function is a univariate regressor"""
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


# Multiple Linear Regression with a UCR dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
diabetes_df = pd.DataFrame(np.c_[diabetes["data"], diabetes["target"]])

print(diabetes_df.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Shape of X_train: {}\nShape of X_test: {}\n".format(X_train.shape, X_test.shape))
print("Shape of y_train: {}\nShape of y_test: {}\n".format(y_train.shape, y_test.shape))


# # evaluating the performance of the SLR
rmse = root_mean_squared_error(y_test, y_pred)
print(
    "RMSE between the predicted and actual diabetes ratings is : {0:.3f}\n".format(rmse)
)

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ss = StandardScaler()
lm = LinearRegression().fit(X_train, y_train)
y_pred = lm.predict(X_test)


df_slr_results = pd.DataFrame(
    {"Actual Value": y_test.flatten(), "Predicted Value": y_pred}
)
print(
    "SKLEARN's linear regression OOTB predictions vs actual diabetes value...\n{}\n".format(
        df_slr_results.head()
    )
)


print(
    "SKLEARN's linear model OOTB scoring on the diabetes dataset: {}\n".format(
        lm.score(X_test, y_test)
    )
)


# CROSS VALIDATION
k_fold = KFold(n_splits=10)
# you can see how the data are split for the k_fold splits
# for train_indices, val_indices in k_fold.split(X_train):
#     print("Train: {} |\n\n Validation: {}\n".format(train_indices, val_indices))


# STORING SCORES OF CROSS VALIDATION
scores = cross_val_score(lm, X_train, y_train, cv=k_fold, scoring='r2')
model_names = [
"model0","model1", "model2", "model3", "model4", 
"model5", "model6", "model7", "model8", "model9"
]
lm_scores_df = pd.DataFrame({"ModelID": model_names, "R^2 Score": scores})
print(lm_scores_df.head(10))

# MODEL COEFFICIENTS OF BEST MODEL
pprint("These are the weights from the linear model\n{}\n".format(lm.coef_))


######################################################################
############### Logistic Regression Classifier #######################
######################################################################
######################################################################


def factorial(n):
    res = 1
    for i in range(1, n + 1):
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


print("\n")
print("This is euler's number: {}".format(e()))
print("This is the real euler's number: {}\n".format(math.e))


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
	logistic error function for Bernoulli/Binary outcome classification
	"""
    j_theta = 0
    for x, y in zip(X, Y):
        j_theta = (
            j_theta + y * math.log(sigmoid(x)) + (1 - y) * math.log(1 - sigmoid(x))
        )
    j_theta = j_theta * (-1 / len(X))
    return j_theta


def logistic_regression(X_train, X_test, Y_train, Y_test):
    """ 
    Logit Link Function which is a univariate regressor extension of the previously 
    built simple linear univariate regression
    """
    prob_a = sigmoid(simple_linear_regression(X_train, X_test, Y_train, Y_test))
    return prob_a


prob_a = logistic_regression(
    dog_heights_train, dog_heights_test, dog_weights_train, dog_weights_test
)
print("This is the output of the logistic regression model:\n{}".format(prob_a))


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Sklearn's Implementation of Logistic Regression #
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Logistic Regression Classification with UCR dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.target_names
iris_df = pd.DataFrame(
    np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

logistic_regressor = LogisticRegression(penalty="l1", warm_start=True).fit(X_train, y_train)

scores = cross_val_score(logistic_regressor, X_train, y_train, cv=10)
print(scores)

# HYPER PARAMETER OPTIMIZATION
param_grid = {
    "C":[0.01, 0.1, 0.5, 0.8, 1.0],
    "penalty":["l1", "l2"],
    "class_weight":[None, "balanced"],
    "warm_start": [True, False]
}

####### Available Methods for Logistic Regressor #######
# Types of fitting
# .fit, .fit_predict, .fit_transform

# Types of scoring
# .predict, .predict_proba, .predict_log_proba, .score

####### Available Attributes for Logistic Regressor #######
# Types of attributes
# .coef_, .intercept_

grid = GridSearchCV(logistic_regressor, param_grid=param_grid, scoring="accuracy", cv=10).fit(X_train, y_train)
y_pred = grid.predict(X_test)
print("These are the predicted labels of the logistic regression classifier\n{}\n".format(y_pred))
prob_y_pred = grid.predict_proba(X_test)
max_prob_y_pred = [max(i) for i in prob_y_pred]
print("These are the predicted probabilities of the logistic regression classifier\n{}\n".format(prob_y_pred))

c_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n{}\n".format(c_mat))

import os
if not os.path.exists("./model_results"):
    os.mkdir("./model_results")

log_reg_results_df = pd.DataFrame(
    {
    "Groud Truth Label": y_test,
    "Predicted Label": y_pred, 
    "Predicted Probability": max_prob_y_pred
    }
)

log_reg_results_df.to_csv(path_or_buf="./model_results/log_reg_results.csv")


plt.figure()
plt.title("Iris Parallel Coordinates Plot")
pd.plotting.parallel_coordinates(iris_df, 'target')
plt.legend()
plt.show()