'''
We create the fibonacci sequence below.
As a refresher, the fibonacci sequence is a recursive sequence in which the last/most-recent term is a sum of the previous two terms
Here instead, we implement MEMOIZATION using built-in python tools to make memoization trivial
'''
from functools import lru_cache # lru cache stands for least-recently used cache

# in order to imbue the powers of the lru_cache you do it as such:
@lru_cache(maxsize=1000) # my default, python will cache the 120 most recently used values
def fibonacci(n):
	# if we put 0 or -1 into the function, we will get an error
	# check to make sure that we input a positive integer
	if type(n) != int:
		raise TypeError("Input must be an integer")
	if n < 1:
		raise ValueError("Use a positive integer")

	# Compute the Nth Term
	if n == 1:
		return 1
	elif n == 2:
		return 1
	elif n > 2:
		return fibonacci(n-1) + fibonacci(n-2)


for n in range(1, 1001):
	print(n, ":", fibonacci(n))

print(fibonacci(3.2))
# print(fibonacci(-1))
# print(fibonacci(0))