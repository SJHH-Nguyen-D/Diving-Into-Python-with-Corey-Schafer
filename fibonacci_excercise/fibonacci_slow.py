'''
We create the fibonacci sequence below.
As a refresher, the fibonacci sequence is a recursive sequence in which the last/most-recent term is a sum of the previous two terms
'''

Here, we define the fibonacci sequence in a manner such that there is poor handling of the recursion and thus the program slows down to basically a halt
def fibonacci(n):
	if n == 1:
		return 1
	elif n == 2:
		return 1
	elif n > 2:
		return fibonacci(n-1) + fibonacci(n-2)

for n in range(1, 101): # start at 1 and stop at 10
	print(n, ":", fibonacci(n))

Here is how to better write the fibonacci sequence such that you do not encounter the above error when you run your recursion 100 times and have to 
run the fibonacci again and again
How we achieve this is through MEMOIZATION, which is the idea that we can store the values for recent function calls so that future calls do not have to repeat the work