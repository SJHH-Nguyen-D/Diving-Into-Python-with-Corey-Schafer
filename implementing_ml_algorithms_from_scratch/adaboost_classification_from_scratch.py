######################################################################
############### Adaboost Classification ##############################
######################################################################

import math
import numpy as np 
from sklearn import datasets
from sklearn.decomposition import PCA

# load in breast cancer dataset for examples
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
names = breast_cancer.target_names


def adaboost_classification():
    pass

print(X)
print(y)
print(names)