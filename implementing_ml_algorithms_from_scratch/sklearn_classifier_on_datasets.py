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
import os
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

from sklearn_classifier_constants import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_model_results_directory():
    if not os.path.exists("./model_results"):
        os.mkdir("./model_results")

#####################################################
################ LOAD DATASETS ######################
#####################################################

from sklearn.datasets import *

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

def write_out_perf_stats():
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

def report_model_report(dataset_df, tree_classifiers):
    """ run all the models at once and report on the results
    TODO: extend this to loop over a set of dataset and classifiers, and takes them as lists for the
    arguments

    TODO: make pipeline object that fits scalers for each classifier that is 
    """
    # list constant

    precision_recall_f1_support = ["Precision", "Recall", "F-Score", "Support"]
    model_performances = []

    for dataset_name, df in dataset_df.items():
        print("##### {} DATASET MODELING #####\n".format(dataset_name.upper()))
        # feature and target splitting
        feature_columns = list(
            df.columns[df.columns != "target"]
        )
        target_col = df.columns[df.columns == "target"]
        X = df[feature_columns].values
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=47
        )

        # modeling and reporting
        for model_category, clf_model in CLASSIFIERS.items():
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


#####################################################
##### CROSS VALIDATING CLASSIFIER PERFORMANCE #######
#####################################################


#####################################################
##### Tabulating Performance of Classifiers #########
#####################################################

model_names = [model for _, model_tup in CLASSIFIERS.items() for model, _ in model_tup.items()]

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
# results_df.to_csv

def report_performances():
    pass

#####################################################
##### Plotting Performance of Classifiers #########
#####################################################
"""
Same as metrics to tabulate
"""

if __name__ == "__main__":
    create_model_results_directory()
    report_model_report(DATASETS, CLASSIFIERS)
