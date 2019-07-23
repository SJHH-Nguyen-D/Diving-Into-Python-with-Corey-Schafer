import numpy as np 
import pandas as pd 
from sklearn.datasets import *

iris = load_iris()
iris = pd.DataFrame(
	np.c_[iris["data"], iris["target"]],
	columns=iris['feature_names']+['target']
)
# print(iris.head())

# print(iris.columns)

def get_item(values):
	return values[4]

iris_feature_columns = list(iris.columns[iris.columns !="target"])
a = iris_feature_columns + ["adams", "orphan"]
b = sorted(a, key=get_item, reverse=True)
print(b)

feature_importances_ = [100, 6000, 1000, 2]


# zip
feature_importance_pair = list(zip(iris_feature_columns, feature_importances_))
# print(feature_importance_pair)

# key
def get_key(feature_importance_pair):
	return feature_importance_pair[1]

# sort
d = sorted(feature_importance_pair, key=get_key, reverse=True)
# print(d)

# print(iris[iris_feature_columns].head())

target_col = iris.columns[iris.columns == "target"]
# print(target_col)

sampleDict = {
              "key1": {"key10": "value10", "key11": "value11"},
              "key2": {"key20": "value20", "key21": "value21"}
              }

# for category, model in sampleDict.items():
# 	for idx, value in model.items():
# 		print(idx)


