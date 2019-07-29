######################################################################
############### Adaboost Classification ##############################
######################################################################

import math
import numpy as np 
from sklearn import datasets
from sklearn.decomposition import PCA
from decision_tree_from_scratch import DecisionTree

# load in breast cancer dataset for examples
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
names = breast_cancer.target_names


class AdaBoostClassifier(DecisionTree):
	def __init__(self, n_features, min_split):
		self.n_features = n_features
		self.min_split = min_split

	def update_weights()
		pass

	def calc_vote_weights()
		pass