######################################################################
############### Support Vector Machine Regressor #####################
######################################################################

import math
import numpy as np 
from sklearn import datasets
from sklearn.decomposition import PCA

# load in iris dataset for examples
cars = datasets.load_boston()
X = cars.data
y = cars.target


# simpler made-up dataset for regression
dog_heights_train = [600, 470, 170, 430, 300]
dog_weights_train = [4.5, 3.5, 11.1, 3.4, 2.3]
dog_heights_test = list(reversed(dog_heights_train))
dog_weights_test = list(reversed(dog_weights_train))
train_dataset = [list(i) for i in zip(dog_heights_train, dog_weights_train)]
test_dataset = [list(i) for i in zip(dog_heights_test, dog_weights_test)]


def svm_regression():
    pass