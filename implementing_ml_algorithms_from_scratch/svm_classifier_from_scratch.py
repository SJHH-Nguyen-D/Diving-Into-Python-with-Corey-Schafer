######################################################################
############### Support Vector Machine Classifier ####################
######################################################################

import math
import numpy as np 
from sklearn import datasets
from sklearn.decomposition import PCA

# load in iris dataset for examples
wine = datasets.load_wine()
X = wine.data
y = wine.target


def svm_classification():
    pass