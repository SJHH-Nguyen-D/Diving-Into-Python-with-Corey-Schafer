'''
We create the fibonacci sequence below.
As a refresher, the fibonacci sequence is a recursive sequence in which the last/most-recent term is a sum of the previous two terms
Here is how to better write the fibonacci sequence such that you do not encounter the above error when you run your recursion 100 times and have to 
run the fibonacci again and again
How we achieve this is through MEMOIZATION, which is the idea that we can store the values for recent function calls so that future calls do not have to repeat the work
'''


# We will first implement MEMOIZATION explicitly
fibonacci_cache = {}
def fibonacci(n):
	# if we have cached the value, then return it
	if n in fibonacci_cache:
		return fibonacci_cache[n]

	# compute the Nth term 
	# We handle this differently than before. We do not simply return the value. 
	# Instead we will first compute the value cache it, and then return it. 
	if n == 1:
		value = 1
	elif n ==2:
		value = 1
	elif n > 2:
		value = fibonacci(n-1) + fibonacci(n-2)

	# then, cache the value into our fibonacci_cache dictionary
	fibonacci_cache[n] = value
	return value

for n in range(1, 1001):
	print(n, ":", fibonacci(n))