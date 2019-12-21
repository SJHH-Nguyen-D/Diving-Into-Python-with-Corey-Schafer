'''
Writing Unit tests for your code in python

This script is a follow along the tutorial of the same name along with this video:
https://www.youtube.com/watch?v=6tNS--WetLI&t=1s

Here, we built unit tests for our code, tear down the tests and learn about best practices.
We will also be using the unittest module that is built-in with Python

instead of having to do print statements everywhere to see if the changes you made to pieces of the code break something
you should develop good unit testing practices to ensure that your code runs smoothly and things make sense at a glance as to what is breaking
The convention to writing a test script is by starting the naming of the script name with "test_"
Let's create a script called test_calc.py where we will build our unit tests to test the code of our calculator

This script will contain all our unittests for our calc.py script
Using the built-in unittests module, we can use it to test our script

The documentation to writiong the assert tests can be viewed here: 
https://docs.python.org/3/library/unittest.html#unittest.TestCase.debug

'''
import unittest
import calc

# we first need to create a class that needs to inherit from unittest.testcase

class TestCalc(unittest.TestCase):

	'''
	Now we need to build functions that do the testing for us.
	The naming convention for use would be to name the tests with test_
	If the methods do not start with the nam test, they will not be run.
	for example, if you name the function add_test, the testing will be skipped
	'''

	def test_add(self):
		# since we inherit from TestCase, we have access to a bunch of asset methods
		# there are a bunch of assert methods from the documentation but we are first 
		# going to start off with using assertEqual to test our add function
		# assertEqual checks to see if a == b
		# here we write our assertEqual method
		# we do this a bunch of times to test a couple of edge cases such as one negative number and one positive number
		######################## ASSERTEQUAL() ###################################

		self.assertEqual(calc.add(10, 2), 12)
		self.assertEqual(calc.add(-1, 1), 0)
		self.assertEqual(calc.add(-1, -1), -2)
		# you might be expecting the console to say that it ran three tests, but it just says it runs 1 because they are all under the same method
		# our goal is not to write as many tests as possible, but it's just to make sure that we write good tests
		# Now let's test the rest of calc functions

	def test_subtract(self):
		# since we inherit from TestCase, we have access to a bunch of asset methods
		# there are a bunch of assert methods from the documentation but we are first 
		# going to start off with using assertEqual to test our add function
		# assertEqual checks to see if a == b
		# here we write our assertEqual method
		# we do this a bunch of times to test a couple of edge cases such as one negative number and one positive number
		self.assertEqual(calc.subtract(10, 2), 12) # purposefully fail this test to see the error it produces
		self.assertEqual(calc.subtract(-1, 1), 0)
		self.assertEqual(calc.subtract(-1, -1), -2)

	def test_multiply(self):
		# since we inherit from TestCase, we have access to a bunch of asset methods
		# there are a bunch of assert methods from the documentation but we are first 
		# going to start off with using assertEqual to test our add function
		# assertEqual checks to see if a == b
		# here we write our assertEqual method
		# we do this a bunch of times to test a couple of edge cases such as one negative number and one positive number
		self.assertEqual(calc.multiply(10, 2), 20)
		self.assertEqual(calc.multiply(-1, 1), -1)
		self.assertEqual(calc.multiply(-1, -1), 1)

	def test_divide(self):
		# since we inherit from TestCase, we have access to a bunch of asset methods
		# there are a bunch of assert methods from the documentation but we are first 
		# going to start off with using assertEqual to test our add function
		# assertEqual checks to see if a == b
		# here we write our assertEqual method
		# we do this a bunch of times to test a couple of edge cases such as one negative number and one positive number
		self.assertEqual(calc.divide(10, 2), 5)
		self.assertEqual(calc.divide(-1, 1), -1)
		self.assertEqual(calc.divide(-1, -1), 1)
		self.assertEqual(calc.divide(5, 2), 2.5)

		######################## ASSERTRAISES() ###################################
		# Check the expectations on our raise ValueError in our divide function
		# do this with the assertRaises function by passing in the Exception that we expect, followed by the function that we want to test
		# then we pass in arguments to the function that we pass into assertRaises
		# notice how we pass in the divide function without any parentheses. If you do, we get an error
		# pass in the arguments that you want to test separately in the next commas
		self.assertRaises(ValueError, calc.divide, 10, 0) # passing a 2 for the 4th argument results in a fail 

		# Most people don't like to test this method of raising exceptions when you have to pass in all the arguments separately.
		# you can see how we can avoid this completely by using a context manager like so:
		# It is the preferred way use context managers when testing Exceptions
		with self.assertRaises(ValueError):
			# call our function normally, like how the function should be called like so:
			calc.divide(10, 0)

		

'''
You might think that you could run the script in terminal or build it in the sublime text editor
With python test_calc.py
Instead, to do that, we need to run unittest as our main module and and then test_calc.py
We can do that by running
>>> python -m unittest test_calc.py

However, we can set it up in the script so that we don't have to run it with the long command above, 
and instead you can run it as: python test_calc.py instead
What it says here is that, if we run that module directly, then run that code within the conditional
When you run the tests, you should get a few dots which indicte the number of tests that are run
If any of the tests should fail, a capital F should indicate 
'''

if __name__ == "__main__":
	unittest.main()