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
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings
import GPyOpt  # for across model global bayesian optimization
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# import models
# singular models & multi-class classifier
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    Perceptron,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


# ensemble models & multi-class classifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)


#####################################################
################ LOAD DATASETS ######################
#####################################################


from sklearn.datasets import *

# Iris Dataset
iris = load_iris()
iris = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)
iris["target"] = iris["target"].astype("int")
print("Iris Dataset")
print(iris.keys())


# Wine Dataset
wine = load_wine()
wine = pd.DataFrame(
    data=np.c_[wine["data"], wine["target"]], columns=wine["feature_names"] + ["target"]
)
wine["target"] = wine["target"].astype("int")
print("Wine Dataset")
print(wine.keys())
print(wine.target.value_counts())


# Breast Cancer Dataset
breast_cancer = load_breast_cancer()
breast_cancer = pd.DataFrame(
    np.c_[breast_cancer["data"], breast_cancer["target"]],
    columns=np.append(breast_cancer["feature_names"], ["target"]),
)
breast_cancer["target"] = breast_cancer["target"].astype("int")
print("Breast cancer Dataset")
print(breast_cancer.keys())


#####################################################
############### PCA, LDA ############################
#####################################################

# PCA and LDA is done for dimensionality reduction
# and data visualization

np.random.seed(0)

# Iris Dataset #
iris_X = iris.iloc[:, :-1].values
iris_Y = iris.iloc[:, -1].values

centers = [[1, 1], [-1, -1], [1, -1]]
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

plt.cla()
pca = PCA(n_components=3)
pca.fit(iris_X)
iris_X = pca.transform(iris_X)

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        iris_X[iris_Y == label, 0].mean(),
        iris_X[iris_Y == label, 1].mean() + 1.5,
        iris_X[iris_Y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
iris_Y = np.choose(iris_Y, [1, 2, 0]).astype(np.float)
ax.scatter(
    iris_X[:, 0],
    iris_X[:, 1],
    iris_X[:, 2],
    c=iris_Y,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

# plt.show()

#################### Wine Dataset ####################
wine_X = wine.iloc[:, :-1].values
wine_Y = wine.iloc[:, -1].values

centers = [[1, 1], [-1, -1], [1, -1]]
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

plt.cla()
pca = PCA(n_components=3)
pca.fit(wine_X)
wine_X = pca.transform(wine_X)

for name, label in [("class_0", 0), ("class_1", 1), ("class_2", 2)]:
    ax.text3D(
        wine_X[wine_Y == label, 0].mean(),
        wine_X[wine_Y == label, 1].mean() + 1.5,
        wine_X[wine_Y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
wine_Y = np.choose(wine_Y, [1, 2, 0]).astype(np.float)
ax.scatter(
    wine_X[:, 0],
    wine_X[:, 1],
    wine_X[:, 2],
    c=wine_Y,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

# plt.show()


#################### Breast Cancer Dataset ####################
# breast_cancer_X = breast_cancer.iloc[:,:-1].values
# breast_cancer_Y = breast_cancer.iloc[:,-1].values

# centers = [[1, 1], [-1, -1], [1, -1]]
# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# plt.cla()
# pca = PCA(n_components=2)
# pca.fit(breast_cancer_X)
# breast_cancer_X = pca.transform(breast_cancer_X)

# for name, label in [('Benign', 0), ('Malignant', 1)]:
#     ax.text3D(breast_cancer_X[breast_cancer_Y == label, 0].mean(),
#               breast_cancer_X[breast_cancer_Y == label, 1].mean() + 1.5,
#               breast_cancer_X[breast_cancer_Y == label, 2].mean(),
#               name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w')
#               )

# # Reorder the labels to have colors matching the cluster results
# breast_cancer_Y = np.choose(breast_cancer_Y, [1, 2, 0]).astype(np.float)

# ax.scatter(
# 	breast_cancer_X[:, 0],
# 	breast_cancer_X[:, 1],
# 	breast_cancer_X[:, 2],
# 	c=breast_cancer_Y,
# 	cmap=plt.cm.nipy_spectral,
#     edgecolor='k'
#     )

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])

# plt.show()

#####################################################
#################### EXPLORE ########################
#####################################################

# wine['quality'].describe()

# filter function
# def is_tasty(wine_quality):
# 	if wine_quality>=7:
# 		return 1
# 	else:
# 		return 0

# wine['tasty'] = wine['quality'].apply(is_tasty)

#####################################################
#################### SPLITTING ######################
#####################################################

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(
    iris.iloc[:, :-1].values, iris["target"].values, test_size=0.3, random_state=47
)

wine_X_train, wine_X_test, wine_y_train, wine_y_test = train_test_split(
    wine.iloc[:, :-1].values, wine["target"].values, test_size=0.3, random_state=47
)

#####################################################
#################### SANDBOX ########################
#####################################################

#@@@@@@@@@@@@@@@@@ IRIS DATASET @@@@@@@@@@@@@@@@@@@@@

# Decision Tree Classifier
decisiontreeclassifier = DecisionTreeClassifier(
    max_depth=5, min_samples_split=10, random_state=123, max_leaf_nodes=5
)

iris_y_pred = decisiontreeclassifier.fit(iris_X_train, iris_y_train).predict(
    iris_X_test
)
decisiontreeclassifier_performance = precision_recall_fscore_support(
    iris_y_test, iris_y_pred
)
decisiontreeclassifier_performance_conf_mat = confusion_matrix(iris_y_test, iris_y_pred)
decisiontreeclassifierscores = accuracy_score(iris_y_test, iris_y_pred)

precision_recall_f1_support = ["Precision", "Recall", "F-Score", "Support"]
print("Decision Tree Performance Chart\n")
for metric, result in zip(precision_recall_f1_support,decisiontreeclassifier_performance):
    print("{}: {}".format(metric, result))
print("\n")

print(
    "Decision Tree Classifier Confusion Matrix\n{}\n".format(
        decisiontreeclassifier_performance_conf_mat
    )
)
print(
    "Decision Tree Classifier Accuracy Score: {}\n".format(decisiontreeclassifierscores)
)

iris_feature_columns = list(iris.columns[iris.columns != "target"])

# Feature Importance Ordering Functions
def get_key(feature_importance_pair):
    """ helper function to sort by highest feature importances """
    return feature_importance_pair[1]

def model_feature_importance(features, treemodel, reverse=True):
    """ prints out the feature importances for a tree model in descending order """
    sorted_featured_importance = sorted(list(zip(features, treemodel.feature_importances_)),
        key=get_key,
        reverse=reverse)
    for feature_importance_pair in sorted_featured_importance:
        print(feature_importance_pair)

print("Decision Tree Feature Importances\n{}\n".format(
    model_feature_importance(iris_feature_columns, decisiontreeclassifier, reverse=True))
)

### Gradient Boosting Classifier ###
gradientboostingclassifier = GradientBoostingClassifier(
    max_depth=5, min_samples_split=10, random_state=123, max_leaf_nodes=5
)

iris_y_pred = gradientboostingclassifier.fit(iris_X_train, iris_y_train).predict(
    iris_X_test
)
gradientboostingclassifier_performance = precision_recall_fscore_support(
    iris_y_test, iris_y_pred
)
gradientboostingclassifier_performance_conf_mat = confusion_matrix(
    iris_y_test, iris_y_pred
)
gradientboostingclassifier_scores = accuracy_score(iris_y_test, iris_y_pred)


print("Gradient Boosting Classifier Importances\n{}\n".format(
    model_feature_importance(iris_feature_columns, gradientboostingclassifier, reverse=True))
)

print("Gradient Boosting Classifier Performance Chart\n")
for metric, result in zip(precision_recall_f1_support, gradientboostingclassifier_performance):
    print("{}: {}".format(metric, result))
print("\n")

print(
    "Gradient Boosting Classifier Confusion Matrix\n{}\n".format(
        gradientboostingclassifier_performance_conf_mat
    )
)
print(
    "Gradient Boosting Classifier Accuracy Score: {}\n".format(
        gradientboostingclassifier_scores
    )
)

### Random Forest Classifier ###
randomforestclassifier = RandomForestClassifier(
    max_depth=5, min_samples_split=10, random_state=123, max_leaf_nodes=5
)

iris_y_pred = randomforestclassifier.fit(iris_X_train, iris_y_train).predict(
    iris_X_test
)
randomforestclassifier_performance = precision_recall_fscore_support(
    iris_y_test, iris_y_pred
)
randomforestclassifier_conf_mat = confusion_matrix(iris_y_test, iris_y_pred)
randomforestclassifier_scores = accuracy_score(iris_y_test, iris_y_pred)


print("Random Forest Classifier Importances\n{}\n".format(
    model_feature_importance(iris_feature_columns, randomforestclassifier, reverse=True))
)


print("Random Forest Classifier Performance Chart\n")
for metric, result in zip(precision_recall_f1_support, randomforestclassifier_performance):
    print("{}: {}".format(metric, result))
print("\n")


print("Random Forest Confusion Matrix\n{}\n".format(randomforestclassifier_conf_mat))
print(
    "Random Forest Classifier Accuracy Score: {}\n".format(
        randomforestclassifier_scores
    )
)

### AdaBoost Classifier ###
adaboostclassifier = AdaBoostClassifier()

iris_y_pred = adaboostclassifier.fit(iris_X_train, iris_y_train).predict(
    iris_X_test
)
adaboostclassifier_performance = precision_recall_fscore_support(
    iris_y_test, iris_y_pred
)
adaboostclassifier_conf_mat = confusion_matrix(iris_y_test, iris_y_pred)
adaboostclassifier_scores = accuracy_score(iris_y_test, iris_y_pred)


print("AdaBoost Classifier Importances\n{}\n".format(
    model_feature_importance(iris_feature_columns, adaboostclassifier, reverse=True))
)

print("AdaBoost Classifier Performance Chart\n")
for metric, result in zip(precision_recall_f1_support, adaboostclassifier_performance):
    print("{}: {}".format(metric, result))
print("\n")


print("AdaBoost Confusion Matrix\n{}\n".format(adaboostclassifier_conf_mat))
print(
    "AdaBoost Classifier Accuracy Score: {}\n".format(
        adaboostclassifier_scores
    )
)

#@@@@@@@@@@@@@@@@@ WINE DATASET @@@@@@@@@@@@@@@@@@@@@

decisiontreeclassifier = DecisionTreeClassifier(
    max_depth=5, min_samples_split=10, random_state=123, max_leaf_nodes=5
)

wine_y_pred = decisiontreeclassifier.fit(wine_X_train, wine_y_train).predict(
    wine_X_test
)
decisiontreeclassifier_performance = precision_recall_fscore_support(
    wine_y_test, wine_y_pred
)
decisiontreeclassifier_performance_conf_mat = confusion_matrix(wine_y_test, wine_y_pred)
decisiontreeclassifierscores = accuracy_score(wine_y_test, wine_y_pred)

pprint(
    "Decision Tree Classifier Performance\nPrecision: {}\nRecall: {}\nF-Score: {}\nSupport: {}\n\n".format(
        decisiontreeclassifier_performance[0],
        decisiontreeclassifier_performance[1],
        decisiontreeclassifier_performance[2],
        decisiontreeclassifier_performance[3],
    )
)
print(
    "Decision Tree Classifier Confusion Matrix\n{}\n".format(
        decisiontreeclassifier_performance_conf_mat
    )
)
print(
    "Decision Tree Classifier Accuracy Score: {}\n".format(decisiontreeclassifierscores)
)

print(decisiontreeclassifier.feature_importances_)

wine_feature_columns = list(wine.columns[wine.columns != "target"])

print("Decision Tree Classifier Feature Importance\n")
for feature, importance in zip(wine_feature_columns, decisiontreeclassifier.feature_importances_):
    print("{}: {}".format(feature,importance))
print("\n")


# Gradient Boosting Classifier
gradientboostingclassifier = GradientBoostingClassifier(
    max_depth=5, min_samples_split=10, random_state=123, max_leaf_nodes=5
)

wine_y_pred = gradientboostingclassifier.fit(wine_X_train, wine_y_train).predict(
    wine_X_test
)
gradientboostingclassifier_performance = precision_recall_fscore_support(
    wine_y_test, wine_y_pred
)
gradientboostingclassifier_performance_conf_mat = confusion_matrix(
    wine_y_test, wine_y_pred
)
gradientboostingclassifier_scores = accuracy_score(wine_y_test, wine_y_pred)

print("Gradient Boosting Classifier Feature Importance\n")
for feature, importance in zip(wine_feature_columns, gradientboostingclassifier.feature_importances_):
    print("{}: {}".format(feature,importance))
print("\n")


pprint(
    "Gradient Boosting Classifier Performance\nPrecision: {}\nRecall: {}\nF-Score: {}\nSupport: {}\n\n".format(
        gradientboostingclassifier_performance[0],
        gradientboostingclassifier_performance[1],
        gradientboostingclassifier_performance[2],
        gradientboostingclassifier_performance[3],
    )
)
print(
    "Gradient Boosting Classifier Confusion Matrix\n{}\n".format(
        gradientboostingclassifier_performance_conf_mat
    )
)
print(
    "Gradient Boosting Classifier Accuracy Score: {}\n".format(
        gradientboostingclassifier_scores
    )
)

# Random Forest Classifier
randomforestclassifier = RandomForestClassifier(
    max_depth=5, min_samples_split=10, random_state=123, max_leaf_nodes=5
)

wine_y_pred = randomforestclassifier.fit(wine_X_train, wine_y_train).predict(
    wine_X_test
)
randomforestclassifier_performance = precision_recall_fscore_support(
    wine_y_test, wine_y_pred
)
randomforestclassifier_conf_mat = confusion_matrix(wine_y_test, wine_y_pred)
randomforestclassifier_scores = accuracy_score(wine_y_test, wine_y_pred)

print("Random Forest Classifier Feature Importance\n")
for feature, importance in zip(wine_feature_columns, randomforestclassifier.feature_importances_):
    print("{}: {}".format(feature,importance))
print("\n")


pprint(
    "Random Forest Classifier Performance\nPrecision: {}\nRecall: {}\nF-Score: {}\nSupport: {}\n\n".format(
        randomforestclassifier_performance[0],
        randomforestclassifier_performance[1],
        randomforestclassifier_performance[2],
        randomforestclassifier_performance[3],
    )
)
print("Random Forest Confusion Matrix\n{}\n".format(randomforestclassifier_conf_mat))
print(
    "Random Forest Classifier Accuracy Score: {}\n".format(
        randomforestclassifier_scores
    )
)

# AdaBoost Classifier
adaboostclassifier = AdaBoostClassifier()

wine_y_pred = adaboostclassifier.fit(wine_X_train, wine_y_train).predict(
    wine_X_test
)
adaboostclassifier_performance = precision_recall_fscore_support(
    wine_y_test, wine_y_pred
)
adaboostclassifier_conf_mat = confusion_matrix(wine_y_test, wine_y_pred)
adaboostclassifier_scores = accuracy_score(wine_y_test, wine_y_pred)

print("AdaBoost Classifier Feature Importance\n")
for feature, importance in zip(wine_feature_columns, adaboostclassifier.feature_importances_):
    print("{}: {}".format(feature,importance))
print("\n")


pprint(
    "AdaBoost Classifier Performance\nPrecision: {}\nRecall: {}\nF-Score: {}\nSupport: {}\n\n".format(
        adaboostclassifier_performance[0],
        adaboostclassifier_performance[1],
        adaboostclassifier_performance[2],
        adaboostclassifier_performance[3],
    )
)
print("AdaBoost Confusion Matrix\n{}\n".format(adaboostclassifier_conf_mat))
print(
    "AdaBoost Classifier Accuracy Score: {}\n".format(
        adaboostclassifier_scores
    )
)


# decisiontreeclassifier attributes
# feature_importances_, max_features, n_features, n_classes, n_outputs, classes_, tree_


#####################################################
#################### PIPELINE #######################
#####################################################

# Preprocessing scaling, normalizing and standardizing
# only on the feature variables

log_reg = LogisticRegression(multi_class="multinomial")
linear_svc = LinearSVC(multi_class="crammer_singer")

# Classifiers that require scaling/normalizationg
"""
SVC
KNN
"""


# Classifiers that are invariant to feature scaling, but might benefit from feature scaling
"""
Naive Bayes
Decision Trees
XGBoost
Random Forest Classifier
Fisher LDA"""

#
# If you need a label binarizer, you can add that in there too
# pipeline = Pipeline(
# {
# "scaler": StandardScaler(),
# "estimator": DecisionTreeClassifier()
# }
# )
# y_pred = pipeline.fit(X_train).predict(X_train))
# y_pred = pipeline.predict(X_test)


#####################################################
##### CROSS VALIDATING CLASSIFIER PERFORMANCE #######
#####################################################


#####################################################
##### Tabulating Performance of Classifiers #########
#####################################################

"""from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    Perceptron,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


# ensemble models & multi-class classifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)"""


model_names = [
    "Logistic Regression",
    "Ridge Classifier",
    "Perception",
    "SGD Classifier",
    "GaussianNB",
    "Decision Tree",
    "K-NN Classifier",
    "Linear SVC",
    "MLP Classifier",
    "Gaussian Process Classifier",
    "Random Forest",
    "Gradient Boosted Classifier",
    "AdaBoost Classifier",
    "Voting Classifier",
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
        "precision": model_names,
        "recall": model_names,
        "f1-score": model_names,
        "runtime": model_names,
        "accuracy": model_names,
        "roc_auc": model_names,
    }
)

# print(results_df.head())


#####################################################
##### Plotting Performance of Classifiers #########
#####################################################
"""
Same as metrics to tabulate
"""
