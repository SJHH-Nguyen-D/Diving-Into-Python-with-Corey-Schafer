""" lag plots can be used to find randomness in data
Typically with lagplots, we want to prime order the data first before we can see how the randomness in the values
"""
import matplotlib.pyplot as plt 
from pandas.plotting import lag_plot
import pandas as pd 
import numpy as np
from sklearn.datasets import *
from datetime import datetime
from timeit import timeit


# Lag plot with Iris Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target


fig, ax = plt.subplots()
for plot_loc, feature in zip(range(411, 415), df.columns[df.columns != 'target']):
	plt.subplot(plot_loc)
	# lag_plot(df[feature].sort_values(ascending=True), label=feature)
	lag_plot(df[feature], label=feature)
	plt.legend(loc='best')
	plt.tight_layout()
plt.show()


# lag plots with time series data
def load_ts_dataset(datasetpath):
	dataset = pd.read_csv(datasetpath)
	dataset['Month'] = pd.to_datetime(dataset["Month"], infer_datetime_format=True)
	indexedDataset = dataset.set_index(["Month"])
	return indexedDataset


# Lag plot with airline passenger time series dataset
airpassengerspath = '../data/AirPassengers.csv'
ts = load_ts_dataset(airpassengerspath)
# t is current time step, t+1 is one future time step
fig, ax = plt.subplots()
# lag_plot(ts.sort_values(by='#Passengers',ascending=True))
lag_plot(ts)
plt.show()


# lag plot with randomly plotted ring dataset
ellipse_data = pd.Series(0.1 * np.random.rand(1000) + 0.9*np.sin(np.linspace(-99*np.pi, 99*np.pi, num=1000)))
fig, ax = plt.subplots()
lag_plot(ellipse_data)
plt.show()

