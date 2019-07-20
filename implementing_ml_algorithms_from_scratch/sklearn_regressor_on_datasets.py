# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Sklearn's Regressor Comparison Sandbox #
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

"""
Sandbox for comparing outofthebox and augmented sklearn regressor
performance on many of sklearn's datasets available through it's API
sklearn.datasets.
"""

#####################################################
################ IMPORT #############################
#####################################################

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import models
# singular models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
stochastic gradient descent regression
perceptron
bayesian regression
OLS regression
ridge regression
lasso regression 

# ensemble models
gradient tree boosting
voting regressor

#####################################################
################ LOAD DATASETS ######################
#####################################################

from sklearn.datasets import *

# Boston dataset
boston = load_boston()
boston = pd.DataFrame(
    boston.data, columns=boston.feature_names
)
print("Boston housing prices Dataset")
print(boston.head())

# diabetes dataset
diabetes = load_diabetes()
diabetes = pd.DataFrame(
    diabetes.data, columns=diabetes.feature_names
)
print("Diabetes Dataset")
print(diabetes.head())

# linnerud dataset
linnerud = load_linnerud()
linnerud = pd.DataFrame(
    linnerud.data, columns=linnerud.feature_names
)
print("Linnerud Dataset")
print(linnerud.head())


#####################################################
#################### PIPELINE #######################
#####################################################

# Preprocessing scaling, normalizing and standardizing
# only on the feature variables


pipeline = Pipeline()


#####################################################
##### CROSS VALIDATING CLASSIFIER PERFORMANCE #######
#####################################################



#####################################################
##### Tabulating Performance of Classifiers #########
#####################################################
"""
Metrics to include:
* run time
* complexity
* hyperparameters/model capacity
* model_name
* accuracy/error
* roc_auc
* sensitivity
* specificity
* TP/TN, FP/FN
* F1 score
"""

#####################################################
##### Plotting Performance of Classifiers #########
#####################################################
"""
Same as metrics to tabulate
"""