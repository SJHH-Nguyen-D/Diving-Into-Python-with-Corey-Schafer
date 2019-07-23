# CONSTANTS for sklearn_classifier_on_datasets.py

# import datasets
from sklearn.datasets import *
import pandas as pd 
import numpy as np

def prepare_datasets():
	# Iris dataset
	iris = load_iris()
	iris = pd.DataFrame(
		np.c_[iris["data"], iris["target"]],
		columns=iris["feature_names"]+["target"]
		)
	iris["target"] = iris["target"].astype("int")

	# Wine dataset
	wine = load_wine()
	wine = pd.DataFrame(
		np.c_[wine["data"], wine["target"]],
		columns=wine["feature_names"]+["target"]
		)
	wine["target"] = wine["target"].astype("int")
	return iris, wine

iris, wine = prepare_datasets()

DATASETS = {"Iris": iris, "Wine": wine}

PRECISION_RECALL_FSCORE_SUPPORT = ["Precision", "Recall", "F-Score", "Support"]

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

# classifier dictionary
CLASSIFIERS = {
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
    }
}