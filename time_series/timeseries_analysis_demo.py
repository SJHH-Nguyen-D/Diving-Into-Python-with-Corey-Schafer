"""
Go through time series analysis example implementing ARIMA

ARIMA stands for:
AR - autogregressive (correlation between current t time and previous lags) with the variable P
I - integrated (deals with integration) and has the variable d, which stands for order of differentiation
MA - moving average (we want to have a moving averages because we never know when we might expect noisy irregular data)
"""

import numpy as np 
import pandas as pd 
import tslearn
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf 
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 10, 6


print(
	"Run the Statistical test for stationarity. The test is called ADF (Augmented Dickey Fuller Test)"\
"Tests whether a unit root is present in the autogregressive time series data model."
"The augmented Dickey-Fuller statistic used in the ADF test is a negative number."\
" The more negative it is, the stronger the rejection of the hypothesis that there "\
"is a unit root. Of course, this is only at some level of confidence. That is to say "\
"that if the ADF test statistic is positive, one can automatically decide not to reject "\
"the null hypothesis of a unit root."
)


def load_dataset(datasetpath):
	dataset = pd.read_csv(datasetpath)
	dataset['Month'] = pd.to_datetime(dataset["Month"], infer_datetime_format=True)
	indexedDataset = dataset.set_index(["Month"])
	return indexedDataset


def plot_log_ts(logts, slidingwindow=12):
	# Now that we tested the stationarity of the data
	# and have discerned that the data are non-stationary, we estimate the trend
	# we take the log because it is the dataset that has been set to montly intervals
	fig, ax = plt.subplots()
	ax.plot(logts)
	ax.set_xlabel('Years')
	ax.set_ylabel('Log Num Passengers')
	ax.set_title('Log Trend of the Passengers')
	plt.show(block=True)
	print("\nWe see that there truly is an uptrend in the data\n")


	# we calculate the moving average for the same window but this time we calculate
	# the moving average and standard deviation for the log of the time series
	log_moving_average = logts.rolling(window=slidingwindow).mean()
	log_moving_std = logts.rolling(window=slidingwindow).std()
	log_moving_var = logts.rolling(window=slidingwindow).var()


	# plot with the log rolling statistics
	fig, ax = plt.subplots()
	ax.set_xlabel('Years')
	ax.set_ylabel('Log Num Passengers')
	ax.set_title('Log Trend of the Passengers')
	ax.plot(logts, label='log trend')
	ax.plot(log_moving_average, color='red', label='log moving mean')
	plt.show()


# calculate the difference between the moving average and actual number of passengers
# so we have mean and actual time series measures. Unless we perform all these transformations
# we will not get a time series that is stationary. We have to get it to a state where it is 
# stationary. This is not a standard way to do all things because it all depends on your time series 
# data and that determines how you can make it stationary. 
# for example, sometimes you might want to take the log, or sometimes you might want to take the square of it
# sometimes cubed roots, so it really just depends on the data that you are working with. 

def log_process_ts(ts):
	logts = np.log(ts)
	logts.dropna(inplace=True)
	return logts


def stationize_timeseries(logts, slidingwindow=12):
	# calculate rolling statistics on log data
	log_moving_average = logts.rolling(window=slidingwindow).mean()
	log_moving_std = logts.rolling(window=slidingwindow).std()
	stationary_log_ts = logts - log_moving_average
	stationary_log_ts.dropna(inplace=True)
	return stationary_log_ts


def test_stationarity(ts, slidingwindow=12):
	""" Entire stationarity test pipeline """

	# plot the time series graph to visually check the stationarity of a univariate time series
	plt.xlabel('Date')
	plt.ylabel('Number of air line passengers')
	plt.plot(ts)
	plt.show()

	# determine rolling statistics
	rolling_average = ts.rolling(window=slidingwindow).mean()
	rolling_std_dev = ts.rolling(window=slidingwindow).std()


	print("\nRolling Mean: {}\nRolling Standard Deviation: {}\n".format(
		rolling_average[11:], rolling_std_dev[11:]
		)
	)

	# plot rolling statistics
	orig = plt.plot(ts, color='blue', label="original")
	mean = plt.plot(rolling_average, color='red', label="rolling mean")
	std = plt.plot(rolling_std_dev, color='green', label='rolling std')
	plt.legend(loc='best')
	plt.title('Rolling Mean and Standard Deviation')
	plt.ylabel("Num of Passengers")
	plt.xlabel("Years")
	plt.show(block=False)


	# Perform Augmented Dickey Fuller Test
	# if your critical value is close to your test statistic, you can see that your data is relatively stationary.
	dftest = adfuller(ts["#Passengers"], autolag='AIC')
	print("\nResult of the Augmented Dickey-Fuller Test: \n{}\n".format(dftest))


	dfoutput = pd.Series(
		dftest[0:4], 
		index=["Test Statistic", "p-value", "# Lags Used", "# Observations Used"]
		)


	# the 4th item in the dftest object is a dictionary of the statistics
	for key, value in dftest[4].items():
		dfoutput["Critical Value (%s)" %key] = value
	print("\n{}\n".format(dfoutput))


	if "f{dfoutput[0]}" > "f{dfoutput[4]}":
		print("Because the test statistic is < the critical value at a p-value of {0:.3f}, "\
			"we fail to reject the null hypthothesis and conclude"\
			" that the data are non-stationary".format(dfoutput[1]))
	else:
		print("Because the test statistic is > the critical value at a p-value of {0:.3f}, we reject the null hypothesis"\
			" and conclude that our data are, in fact, stationary".format(dfoutput[1]))



# calculate the weighted average of the times series
# this is to see the trend within the time series
ts = load_dataset('../data/AirPassengers.csv')
# stationized_ts = stationize_timeseries(log_process_ts(ts))
log_ts = np.log(ts)
exponentialdecayweightedaverage = log_ts.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(log_ts, color='black', label='stationized time series')
plt.plot(exponentialdecayweightedaverage, color='red', label='weighted average')
plt.show()
print("\nFrom what we can tell from the plot, as the time series is progressing, the average of the" \
	" time series is also uptrending with respect to time.")


# Another transformation is to take the log scaled data and subtract the weighted average from it
# we do this again as as test to determine whether our is in a state of stationarity
datasetlogscaleminusmovingexponentialdecayaverage = log_ts - exponentialdecayweightedaverage
datasetlogscaleminusmovingexponentialdecayaverage.dropna(inplace=True) # there are some infinite or na values
test_stationarity(datasetlogscaleminusmovingexponentialdecayaverage)
print("\nJudging by the flat trend of the rolling mean and standard deviation of the weighted average,"\
	" we can conclude that our weighted mean data have no trend and that it is stationary.\n")

# The other transformation we can do is to shift the time series data
# we do this so that we can perform time series forecasting
# we do this by subtracting the log of the time series by a shift of the log of the time series
# Here we have just shifted the values by 1 or taken a lag of 1
# from this, we see that the integration portion of the ARIMA model, d (the order of integration) is now 1
datasetlogdiffshifting = log_ts - log_ts.shift()
datasetlogdiffshifting.dropna(inplace=True)
fig, ax = plt.subplots()
ax.plot(datasetlogdiffshifting)
plt.show()

test_stationarity(datasetlogdiffshifting)
print("\nJudging by the flat trend of the rolling mean and standard deviation of the weighted average,"\
	" we can conclude that our weighted mean data have no trend and that it is stationary.\n")

# Now we decompose our time series for analysis
decomposition = seasonal_decompose(log_ts)
trend = decomposition.trend
seasonal = decomposition.seasonal
# the residuals just plot the irregularities in your data
residual = decomposition.resid 

ts_decomps = [log_ts, trend, seasonal, residual]
plot_label_list = ['original', 'trend', 'seasonal', 'residual']

fig, ax = plt.subplots()
for plot_loc, decomp, plot_label in zip(range(411, 415), ts_decomps, plot_label_list):
	plt.subplot(plot_loc)
	plt.plot(decomp, label=plot_label)
	plt.legend(loc='best')
	plt.tight_layout()
plt.show()
print("\nFrom what we can see from the plot, we can tell that the data is uptrending, with seasonality")

# look at the noise from the data and check if the noise is stationary or not
decomposedlogdata = residual
decomposedlogdata.dropna(inplace=True)
test_stationarity(decomposedlogdata)
print("\nVisually, from the output of the graph, we see that the residuals of the log of the time series is not stationary."\
	" That is why we have to have your moving average parameter in place so that it smooths and setup to predict what will happen next.")

# Now that we know the value of d, the Integration parameter of the ARIMA model, which is the order of differentiation
# But how can you know the values of P and Q, which are the Autoregresive lag correlations and Moving Average, respectively?

# To do that we have to plot ACF and PACF plots, which stand for autocorrelation fuction and partial correlation function
# To calculate the value of Q, we need the ACF graph, and the value of P, we need the PACF graph

lag_acf = acf(datasetlogdiffshifting, nlags=20)
lag_pacf = pacf(datasetlogdiffshifting, nlags=20, method='ols')


# Plot ACF to determine the Q(Moving Average part of ARIMA)
fig, ax = plt.subplots()
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetlogdiffshifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetlogdiffshifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')


# Plot PACF to determine the P(Autoregressive part of ARIMA)
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetlogdiffshifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetlogdiffshifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')

plt.tight_layout()
plt.show()
print("\nFrom these graphs, to determine the P and Q values, we must look at values on the x-axis where"\
	"the graph first hits the 0 mark long the y-axis values"\
	"What we can gleen here is that the values for P and Q are 2.0."\
	" So we have the values of P/Q/d: 2.0/2.0/1.0, respectively. And we can substitute these values into the ARIMA model\n")

# setting up the parameters as we have come to know them
AR = 2 # P
I = 1 # d
MA = 2 # Q

# AutoRegressive Integrated Moving Average (ARIMA) model
fig, ax = plt.subplots()
model = ARIMA(log_ts, order=(AR, I, MA))
results_ARIMA = model.fit(disp=1)
plt.plot(datasetlogdiffshifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
rss = sum((results_ARIMA.fittedvalues-datasetlogdiffshifting['#Passengers'])**2)
plt.title('Residual Sum of Squares: {0:.4f}'.format(rss))
print("\n####### Plotting AR Model ########")
plt.show()
print("From the title of the plot, we can see that the RSS is fairly low, meaning that is a good fit for prediction")

# Moving Average (MA) Model 
AR = 0 # P
I = 1 # d
MA = 2 # Q

fig, ax = plt.subplots()
model = ARIMA(log_ts, order=(AR, I, MA))
results_AR = model.fit(disp=1)
plt.plot(datasetlogdiffshifting)
plt.plot(results_AR.fittedvalues, color='red')
rss = sum((results_AR.fittedvalues-datasetlogdiffshifting['#Passengers'])**2)
plt.title('Residual Sum of Squares: {0:.4f}'.format(rss))
print("\n####### Plotting AR Model ########")
plt.show()


# Fit them all into a combined ARIMA model
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())


# Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


# predictions for the fitted values
predictions_ARIMA_log = pd.Series(log_ts['#Passengers'].ix[0], index=log_ts.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
print(predictions_ARIMA_log.head())

# After we have the predictions for the data, we need to transform it back to the original form by taking the exponential of it again
predictions_ARIMA = np.exp(predictions_ARIMA_log)
print(predictions_ARIMA.head())
fig, ax = plt.subplots()
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.legend(loc='best')
plt.show()
print("\nAs we can see from the visualization, we have captured the shape, seasonality, trend of the data, and only the magnitude has changed")

# we pass in the first index of the time series, and how many data points you wanted that time series for. 
# In this case, we had 144 observations, plus the 120 extra months for prediction = 264. 
results_ARIMA.plot_predict(1, 264)
x = results_ARIMA.forecast(steps=120) # 10 years
print(x[1])
print("x[1] has {} predictions".format(len(x[1])))
print("The exponential of these predictions is: \n{}\n".format(np.exp(x[1])))



# def main():
# 	ts = load_dataset('./data/AirPassengers.csv')
# 	stationized_ts = stationize_timeseries(log_process_ts(ts))
# 	test_stationarity(stationized_ts)


# if __name__ == "__main__":
# 	main()


# 2506