from sklearn.datasets import *
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def average(x):
    """ x is a collection of numbers """
    return sum(x) / len(x)


def standardize(x):
    """ we center the data around mean 0, std dev 1 """
    a = math.sqrt(b ** 2 + c ** 2)
    a = a / a
    b = b / a
    c = c / a
    return a, b, c


def svd(x):
    return


def eigenvalues(distances):
    eigenvalues = sum(map(lambda x: x ** 2, distances))
    return eigenvalues


def pc_variance(eigenvalues, n_samples):
    pc_variance = eigenvalues / (n_samples - 1)
    return pc_variance


def total_variance(pc_variances):
    return sum(pc_variances)


def scree_plot(data, distances):
    """ plots the variance contribution that each principal component accounts for
	in terms of the total amount of variance """
    for i in distances:
        ev = eigenvalues(i)
        pc_var_contribution = pc_variance(ev) / total_variance(pc_variances)
        plt.plot(pc_var_contribution)

    plt.show()


# load data
data = load_iris()
df = pd.DataFrame(
    np.c_[data["data"], data["target"]], columns=data["feature_names"] + ["target"]
)

feature_columns = df.columns[df.columns != "target"]
df = df[feature_columns]

# prep pca
ss = StandardScaler()
scaled_data = ss.fit_transform(df)
print(type(scaled_data))
print(scaled_data)


# max allowed PCs is the minimum between num_features and num_obs
pca = PCA(n_components=min(df.shape[0], df.shape[1]))

# calculate loading scores and the variation each PC accounts for
pca.fit(scaled_data)

# generate coordinates for a PCA graph based on the loading scores and the scaled data
# loading scores * scaled data
pca_data = pca.transform(scaled_data)

# before we do a scree plot, we calculate the percentage of variation that each principal component accounts for
pct_variation = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

# prepare labels for the PCA graph
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

# print the loading scores
print(loading_scores[top_features])
print("************")
print(sorted_loading_scores)
print(type(feature_columns))
