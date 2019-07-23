######################################################################
############### Support Vector Machine Classifier ####################
######################################################################

import math
import numpy as np 
import pandas as pd 
from sklearn import datasets
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("ggplot")

# load in iris dataset for examples
wine = datasets.load_wine()
wine = pd.DataFrame(np.c_[wine["data"], wine["target"]],
	columns=wine["feature_names"]+["target"])
X = wine[wine.columns[wine.columns != "target"]].values
y = wine[wine.columns[wine.columns == "target"]].values

print(wine.head())

class Support_Vector_Machine:
	def __init__(self, C=0.01, visualization=True):
		self.visualization = visualization
		self.colors {1: "r", -1: "b"}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = plt.fig.add_subplot(1, 1, 1)

	def fit(self, X, y):
		self.w = 
		self.b = 

	def predict(self, X):
		# sign(x dot w+b)
		y_pred = np.sign(np.dot(np.array(X), self.w) + self.b)
		return y_pred
