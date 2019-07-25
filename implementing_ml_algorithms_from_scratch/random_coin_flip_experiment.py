# simulating a random coin flip experiment and then plotting the distribution of this outcome
# to a probability distribution

import random
from pprint import pprint

def flip(p):
	""" 
	flips a coin giving a threshold value of a coin.
	If the randomly generated value is less than the assigned threshold value,
	then it is considered a coin, else heads.
	"""
	return "H" if random.random() < p else "T"

random.seed(0)

N = 100
flip_threshold = 0.5

flip_experiments = [flip(flip_threshold) for i in range(N)]
pprint("This is the results of the first distribution:\n{}\n".format(flip_experiments))

import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
ax.hist(flip_experiments)
ax.set(xlabel="Coin Flip Results", ylabel="Frequency Counts of Each Flip", title="Result of {} coin flips experiment".format(N))
plt.show()

M = 50
flip_experiments_2 = [flip(flip_threshold) for i in range(M)]
print("This is the results of the second flip experiment:\n{}\n".format(flip_experiments_2))


fig, ax = plt.subplots()
ax.hist(flip_experiments_2)
ax.set(xlabel="Coin Flip Results", ylabel="Frequency Counts of Each Flip", title="Result of 50 coin flips experiment 2")
plt.show()


# for a 3 coin flip experiment, we want to then plot the probabiltiy distribution for each set of possibilities for this experiment:
"""
set of possible outcomes:
HHH, HHT, HTH, HTT, THH, THT, TTH, TTT
"""
# let random variable X be the number of heads after 3 flips of a fair coin.

# What is the probabilty that our experiment, for DRV X=0 results in 0 heads? it is 1/8

# what is the probability that our random variable X, the number of heads after 3 coin flips =1. 3/8

# what is the probability that our discrete random variable X = 2. 3/8

## what is the probabiltiy that our DRV X = 3. 1/8

############### let's plot this probabiltiy distribution #####################

#@@@@@@@@@@@@ Using just the regular histogram plotting function @@@@@@@@@@@@@
# possible values for DRV X, which is again, the number of heads in a 3 coin flip experiment
X = [0, 1, 2, 3]

# probabilities for each probability of X from working out the fractional probabilities above
probabilities = [1/8, 3/8, 3/8, 1/8]

fig, ax = plt.subplots()
ax.plot(X, probabilities)
ax.set(ylabel="probabilty", xlabel="Value for X", title="Coin Flips Probability Distribution")
plt.show()


#@@@@@@@@@@@@ Using the norm function from scipy @@@@@@@@@@@@@
from scipy.stats import bernoulli
import seaborn as sb

data_bern = bernoulli.rvs(size=1000,p=flip_threshold)
ax = sb.distplot(data_bern,
                  kde=True,
                  color='crimson',
                  hist_kws={"linewidth": 25,'alpha':1})
ax.set(xlabel='Bernouli', ylabel='Frequency')
plt.show()

