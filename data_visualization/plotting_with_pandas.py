from sklearn.datasets import *

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.keys())

ss = StandardScaler(copy=True, with_mean=True, with_std=True)

ss.fit(df)

print((ss.mean_[:-1]))

scaled_df = pd.DataFrame(ss.transform(df), columns=data.feature_names + ['target'])

print(df.head())
print(scaled_df.head())

# scatter, hist, bar, barh, hexbin, pie, kde

scaled_df.plot(x='petal width (cm)', kind='kde')
scaled_df.plot(x='petal width (cm)', y='petal length (cm)', kind='scatter')

plt.show()

