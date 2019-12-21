'''
This is an example script that contains our fictional create calculator application which has a few calulator functions.
The tutorial for this script comes from the unittesting_your_code.py
'''

def add(x, y):
	'''Add Function '''
	return x + y

def subtract(x, y):
	''' Subtraction Function '''
	return x - y

def multiply(x, y):
	''' Multiplication Function '''
	return x * y

def divide(x, y):
	''' Return division function '''
	if y == 0:
		# here, we want to test that our expectations are running fine here.
		raise ValueError("One does not simply divide by zero")
	
	# we can also try and change our code
	# In this example we try to have the divide return a floor division
	# floor division function is just division which does not return the remainder
	# floor division can be notated as a double division sign
	# if our division doesn't return a remainder
	# Therefore there are no decimals returned, only whole numbers, then our code runs successfully without trouble
	# however if there is a remainder from our floor division, we 
	# return x // y # we run the test and realize that we are running floor division instead of regular division so we comment it out and change it back.
	return x / y

