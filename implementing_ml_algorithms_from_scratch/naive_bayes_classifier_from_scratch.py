######################################################################
############### Naive Bayes Classifier ###############################
######################################################################

import math
import numpy as np 
from sklearn import datasets
from sklearn.decomposition import PCA

# load in iris dataset for examples
iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.target_names


def naive_bayes_classification():
    pass