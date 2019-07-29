lfrom sklearn.datasets import *
import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler


def average(x):
	""" x is a collection of numbers """
	return sum(x)/len(x)

def standardize(x):
	""" we center the data around mean 0, std dev 1 """
	a = math.sqrt(b**2 + c**2)
	a = a/a
	b = b/a
	c = c/a
	return a, b, c

def svd(x):
	return

def eigenvalues(distances):
	eigenvalues = sum(map(lambda x: x**2, distances))
	return eigenvalues

def pc_variance(eigenvalues, n_samples):
	pc_variance = eigenvalues/(n_samples-1)
	return pc_variance

def total_variance(pc_variances):
	return sum(pc_variances)

def scree_plot(data, distances):
	""" plots the variance contribution that each principal component accounts for
	in terms of the total amount of variance """
	for i in distances:
		ev = eigenvalues(i)
		pc_var_contribution = pc_variance(ev)/total_variance(pc_variances)
		plt.plot(pc_var_contribution)

	plt.show()


# load data
data = load_iris()
df = pd.DataFrame(np.c_[data["data"], data["target"]],
	columns=data["feature_names"] + ["target"])

# prep pca
ss = StandardScaler()
scaled_data = ss.fit_transform(df)
print(type(scaled_data))
print(scaled_data)


# run pca
