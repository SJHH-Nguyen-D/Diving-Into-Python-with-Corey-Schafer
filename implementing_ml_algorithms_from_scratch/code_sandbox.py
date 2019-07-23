import numpy as np 
import pandas as pd 
from sklearn.datasets import *

iris = load_iris()
X = iris.data 
y = iris.target 
iris = pd.DataFrame(
	np.c_[iris["data"], iris["target"]],
	columns=iris['feature_names']+['target']
)

beeb = np.c_[X, y]
print(beeb)

# def get_item(values):
# 	return values[4]

# iris_feature_columns = list(iris.columns[iris.columns !="target"])
# a = iris_feature_columns + ["adams", "orphan"]
# b = sorted(a, key=get_item, reverse=True)
# print(b)
