# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Sklearn's Classifier Comparison Sandbox #
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

"""
Sandbox for comparing outofthebox and augmented sklearn classifier
performance on many of sklearn's datasets available through it's API
sklearn.datasets.
"""
#####################################################
################ IMPORT #############################
#####################################################

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
 

# import models
# singular models & multi-class classifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC 
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


# ensemble models & multi-class classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier


#####################################################
################ LOAD DATASETS ######################
#####################################################


from sklearn.datasets import *

# Iris Dataset
iris = load_iris()
iris = pd.DataFrame(
    iris.data, columns=iris.feature_names
)
print("Iris Dataset")
print(iris.head())

plt.figure()
plt.title("Iris Parallel Coordinates Plot")
pd.plotting.parallel_coordinates(iris, 'target')
plt.show()

# Wine Dataset
wine = load_wine()
wine = pd.DataFrame(
    wine.data, columns=wine.feature_names
)
print("Wine Dataset")
print(wine.head())

plt.figure()
plt.title("wine Parallel Coordinates Plot")
pd.plotting.parallel_coordinates(wine, 'target')
plt.show()

# Breast Cancer Dataset
breast_cancer = load_breast_cancer()
breast_cancer = pd.DataFrame(
    breast_cancer.data, columns=breast_cancer.feature_names
)
print("Breast cancer Dataset")
print(breast_cancer.head())

plt.figure()
plt.title("Iris Parallel Coordinates Plot")
pd.plotting.parallel_coordinates(iris_df, 'target')
plt.show()

# Digits Dataset
digits = load_digits()
digits = pd.DataFrame(digits.data)
print("Digits Dataset")
print(digits.head())


#####################################################
#################### PIPELINE #######################
#####################################################

# Preprocessing scaling, normalizing and standardizing
# only on the feature variables

log_reg = LogisticRegression(multi_class="multinomial")
linear_svc = LinearSVC(multi_class="crammer_singer")

# pipeline = Pipeline()


#####################################################
##### CROSS VALIDATING CLASSIFIER PERFORMANCE #######
#####################################################



#####################################################
##### Tabulating Performance of Classifiers #########
#####################################################
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC 
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


# ensemble models & multi-class classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier


model_names = [
"Logistic Regression", "RidgeClassifier", "Perception", "SGDClassifier", 
"GaussianNB", "Decision Tree", "K-NN Classifier", "Linear SVC", "MLP Classifier", 
"Gaussian Process Classifier", "Random Forest", "Gradient Boosted Classifier", 
"AdaBoost Classifier", "Voting Classifier"
]

dataset_names = ["Iris", "Wine", "Breast Cancer", "Digits"]

results_df = pd.DataFrame(
	{
	"model_name": model_names,
	"dataset_name": model_names,
	"num_params": model_names,
	"best_params": model_names,
	"sensitivity": model_names,
	"specificity": model_names,
	"precision":model_names,
	"recall": model_names,
	"f1-score": model_names,
	"runtime":model_names,
	"accuracy":model_names,
	"roc_auc":model_names,
	}
)

print(results_df.head())


#####################################################
##### Plotting Performance of Classifiers #########
#####################################################
"""
Same as metrics to tabulate
"""