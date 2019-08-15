from sklearn.datasets import *
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

wine = load_wine()
cancer = load_breast_cancer()
boston = load_boston()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = pd.Series(wine.target)
feature_columns = df.columns[df.columns != "target"]

df = df[feature_columns]

# prep pca with a standard scaler to centre the data around the origin
ss = StandardScaler()
scaled_data = ss.fit_transform(df)


# fit transform PCA object
pca = PCA(n_components=min(df.shape[0], df.shape[1]))
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

pct_variation = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

labels = ["PC{}".format(x) for x in range(1, len(pct_variation) + 1)]

# run scree plot to see how many principal components should go into the final plot
# select the PCs that describe the most amount of variation in the data
fig, ax = plt.subplots()
ax.bar(x=range(1, len(pct_variation) + 1), height=pct_variation, tick_label=labels)
ax.set(xlabel="Principal Components", ylabel="% Variation", title="Scree Plot of PCA")
plt.show()

# from the PCA plot, we can use the information we learned from the scree plot
# first we put the new coordinates, created by the pca.transform(scaled_data) operation, into a nice matrix
# where the rows have sample labels and the solumns have the PCA labels
pca_df = pd.DataFrame(pca_data, columns=labels)
print(pca_df.head())

# plot using the PCA dataframe
plt.scatter(pca_df.PC1, pca_df.PC1)
plt.title("My PCA Graph")
plt.xlabel("PC1 - {}%".format(pct_variation[0]))
plt.ylabel("PC1 - {}%".format(pct_variation[1]))

# this loop allows us to annotate (put) the sample names to the graph
for sample in pca_df.index:
    plt.annotate(sample, (pca_df["PC1"].loc[sample], pca_df["PC2"].loc[sample]))

plt.show()

# Now let's take a look at the loading scores for PC1 to see which features have the largest influence on separating the clusters along the X-axis
# principal components are 0-indexed, so PC1 is at index 0.
loading_scores = pd.Series(pca.components_[0], index=feature_columns)

# sort the loading scores based on their magnitude of influence (absolute value, as some of the loading scores can have a negative value)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# get top features as a mask criteria for our dataframe
top_features = sorted_loading_scores[:4].index.values

print(sorted_loading_scores)

print(sorted_loading_scores[top_features])

a = np.round(pca.explained_variance_ratio_ *100, decimals=2)
b = pca.components_
print("Eigenvalues for PC1\n{}\n".format(b[0]))
