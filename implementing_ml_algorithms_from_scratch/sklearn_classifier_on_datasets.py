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
######### SPLITTING, MODELING, REPORTING ############
#####################################################


precision_recall_f1_support = ["Precision", "Recall", "F-Score", "Support"]

datasets = {"Iris": iris, "Wine": wine}

classifiers = {
    "tree": {
        "Decision Tree Classifier": DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, random_state=123, max_leaf_nodes=5
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            max_depth=5, min_samples_split=10, random_state=123, max_leaf_nodes=5
        ),
        "Random Forest Classifier": RandomForestClassifier(
            max_depth=5, min_samples_split=10, random_state=123, max_leaf_nodes=5
        ),
        "AdaBoost Classifier": AdaBoostClassifier(),
    },
    "linear_model": {
        "Logistic Regression": LogisticRegression(),
        "Ridge Classifier": RidgeClassifier(),
        "Perceptron": Perceptron(),
        "SGD Classifier": SGDClassifier(),
    },
    "probabilistic": {"Gaussian Naive Bayes": GaussianNB(), "Bernoulli Naive Bayes": BernoulliNB()},
    "distance_based": {
        "SVC": LinearSVC(),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=3),
    },
}

# Feature Importance Ordering Functions
def get_key(feature_importance_pair):
    """ helper function to sort by highest feature importances """
    return feature_importance_pair[1]


def model_feature_importance(feature_columns, treemodel, reverse=True):
    """ prints out the feature importances for a tree model in descending order """
    tup_list = list(zip(feature_columns, treemodel.feature_importances_))
    sorted_featured_importance = sorted(tup_list, key=get_key, reverse=reverse)
    for feature_importance_pair in sorted_featured_importance:
        print(feature_importance_pair)


def report_model_report(dataset_df, tree_classifiers):
    """ run all the models at once and report on the results
    TODO: extend this to loop over a set of dataset and classifiers, and takes them as lists for the
    arguments

    TODO: make pipeline object that fits scalers for each classifier that is 
    """
    # list constant
    precision_recall_f1_support = ["Precision", "Recall", "F-Score", "Support"]

    for dataset_name, dataset_dataframe in dataset_df.items():
        print("##### {} DATASET MODELING #####\n".format(dataset_name.upper()))
        # feature and target splitting
        feature_columns = list(
            dataset_dataframe.columns[dataset_dataframe.columns != "target"]
        )
        target_col = dataset_dataframe.columns[dataset_dataframe.columns == "target"]
        X = dataset_dataframe[feature_columns].values
        y = dataset_dataframe[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=47
        )

        # modeling and reporting
        for model_category, clf_model in classifiers.items():
            # Tree-Based Models
            if model_category == "tree":
                print("##### {} MODELS #####\n".format(model_category.upper()))
                for clf_name, model in clf_model.items():
                    print("##### {} Performance report on the {} Dataset #####\n".format(clf_name, dataset_name))

                    # fit and make inference
                    y_pred = model.fit(X_train, y_train).predict(X_test)

                    # reporting
                    print("{} Performance Chart\n".format(clf_name))
                    for metric, result in zip(
                        precision_recall_f1_support,
                        precision_recall_fscore_support(y_test, y_pred),
                    ):
                        print("{}: {}".format(metric, result))
                    print("\n")

                    print(
                        "{} Confusion Matrix\n{}\n".format(
                            clf_name, confusion_matrix(y_test, y_pred)
                        )
                    )

                    print(
                        "{} Accuracy Score: {}\n".format(
                            clf_name, accuracy_score(y_test, y_pred)
                        )
                    )

                    print("{} Feature Importance\n".format(clf_name))
                    print(model_feature_importance(feature_columns, model, reverse=True))
                    print("\n")

            # Distance Based Models
            elif model_category == "distance_based":
                print("##### {} MODELS #####\n".format(model_category.upper()))
                for clf_name, model in clf_model.items():
                    print("##### {} Performance report on the {} Dataset #####\n".format(clf_name, dataset_name))

                    # fit and make inference
                    y_pred = model.fit(X_train, y_train).predict(X_test)

                    # reporting
                    print("{} Performance Chart\n".format(clf_name))
                    for metric, result in zip(
                        precision_recall_f1_support,
                        precision_recall_fscore_support(y_test, y_pred),
                    ):
                        print("{}: {}".format(metric, result))
                    print("\n")

                    print(
                        "{} Confusion Matrix\n{}\n".format(
                            clf_name, confusion_matrix(y_test, y_pred)
                        )
                    )

                    print(
                        "{} Accuracy Score: {}\n".format(
                            clf_name, accuracy_score(y_test, y_pred)
                        )
                    )

            # Linear Models
            elif model_category == "linear_model":
                print("##### {} MODELS #####\n".format(model_category.upper()))
                for clf_name, model in clf_model.items():
                    print("##### {} Performance report on the {} Dataset #####\n".format(clf_name, dataset_name))

                    # fit and make inference
                    y_pred = model.fit(X_train, y_train).predict(X_test)

                    # reporting
                    print("{} Performance Chart\n".format(clf_name))
                    for metric, result in zip(
                        precision_recall_f1_support,
                        precision_recall_fscore_support(y_test, y_pred),
                    ):
                        print("{}: {}".format(metric, result))
                    print("\n")

                    print(
                        "{} Confusion Matrix\n{}\n".format(
                            clf_name, confusion_matrix(y_test, y_pred)
                        )
                    )

                    print(
                        "{} Accuracy Score: {}\n".format(
                            clf_name, accuracy_score(y_test, y_pred)
                        )
                    )
            # Probabilistic  Models
            elif model_category == "probabilistic":
                print("##### {} MODELS #####\n".format(model_category.upper()))
                for clf_name, model in clf_model.items():
                    print("##### {} Performance report on the {} Dataset #####\n".format(clf_name, dataset_name))

                    # fit and make inference
                    y_pred = model.fit(X_train, y_train).predict(X_test)

                    # reporting
                    print("{} Performance Chart\n".format(clf_name))
                    for metric, result in zip(
                        precision_recall_f1_support,
                        precision_recall_fscore_support(y_test, y_pred),
                    ):
                        print("{}: {}".format(metric, result))
                    print("\n")

                    print(
                        "{} Confusion Matrix\n{}\n".format(
                            clf_name, confusion_matrix(y_test, y_pred)
                        )
                    )

                    print(
                        "{} Accuracy Score: {}\n".format(
                            clf_name, accuracy_score(y_test, y_pred)
                        )
                    )

"""
NOTES:

decisiontreeclassifier attributes
feature_importances_, max_features, n_features, n_classes, n_outputs, classes_, tree_

"""

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

if __name__ == "__main__":
    report_model_report(datasets, classifiers)