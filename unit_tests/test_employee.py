import unittest
from employee import Employee
<<<<<<< HEAD
from unittest.mock import patch # we can use patch as a decorator or as a context manager

# patch allows us to mock an object during a test and then that object is automatically restored after the test is run.

=======
from unittest.mock import patch # can be used as a context manager or a decorator
>>>>>>> 127c6c57686d95ac80fb811aa750727e3a136371

class TestEmployee(unittest.TestCase):
	# class methods allow us to work with the class itself, rather than the instance of the class
	# class methods are denoted with the @clasmethod decorator before the method definition
	@classmethod
	def setUpClass(cls):
		print("setUpClass")

<<<<<<< HEAD
=======
	# class methods setUpClass and tearDownClass to put up instances before and after testing
	@classmethod
	def setUpClass(cls):
		print("setUpClass")

>>>>>>> 127c6c57686d95ac80fb811aa750727e3a136371
	@classmethod
	def tearDownClass(cls):
		print("tearDownClass")

<<<<<<< HEAD
	# using the setUp method to keep code DRY
	# The setUp() method will run its code BEFORE every single test
	# In order to access these instances throuhgout our script, we will have to 
	# set these variables as instance attributes by putting self.variable
	# since those are now instance attributes, because they have self. infront of them, 
	# we have to add self to the beginning when referring to them
	def setUp(self):
		# These two employee instance attributes are created before every single one of our tests
		# They will be created to be used as testing examples
		print("setUp")
		self.emp_1 = Employee("Corey", "Schafer", 50_000)
		self.emp_2 = Employee("Sue", "Smith", 60_000)

	# use the tearDown() method to keep code DRY
	# the tearDown() method will run its code AFTER every single test
	def tearDown(self):
		print("tearDown\n")
		pass

=======
	# This setUp method will do something before the testing starts, which keeps our code DRYer
	def setUp(self):
		print("setUp")
		self.emp_1 = Employee("Corey", "Schafer", 50_000)
		self.emp_2 = Employee("Sue", "Smith", 60_000)

	def tearDown(self):
		print("tearDown\n")

>>>>>>> 127c6c57686d95ac80fb811aa750727e3a136371
	def test_email(self):
		print("test_email")
		# we test if those assertions on those new instances hold true
		self.assertEqual(self.emp_1.email, "Corey.Schafer@email.com")
		self.assertEqual(self.emp_2.email, "Sue.Smith@email.com")

		# we change the values for the names
		self.emp_1.first = "John"
		self.emp_2.first = "Jane"

		# we check if given the assertions that the values still hold true
		self.assertEqual(self.emp_1.email, "John.Schafer@email.com")
		self.assertEqual(self.emp_2.email, "Jane.Smith@email.com")
<<<<<<< HEAD

	def test_fullname(self):
		print("test_fullname")
		self.assertEqual(self.emp_1.fullname, "Corey Schafer")
		self.assertEqual(self.emp_2.fullname, "Sue Smith")

		self.emp_1.first = "John"
		self.emp_2.first = "Jane"

		self.assertEqual(self.emp_1.fullname, "John Schafer")
		self.assertEqual(self.emp_2.fullname, "Jane Smith")

	def test_apply_raise(self):
		print("test_apply_raise")
		self.emp_1.apply_raise()
		self.emp_2.apply_raise()

		self.assertEqual(self.emp_1.pay, 52_500)
		self.assertEqual(self.emp_2.pay, 63_500)
=======

	def test_fullname(self):
		print("test_fullename")
		self.assertEqual(self.emp_1.fullname, "Corey Schafer")
		self.assertEqual(self.emp_2.fullname, "Sue Smith")

		self.emp_1.first = "John"
		self.emp_2.first = "Jane"

		self.assertEqual(self.emp_1.first, "John Schafer")
		self.assertEqual(self.emp_2.first, "Jane Smith")

	def test_apply_raise(self):
		print("test_apply_raise")

		self.emp_1.apply_raise()
		self.emp_2.apply_raise()

		self.assertEqual(self.emp_1.pay, 52_500)
		self.assertEqual(self.emp_2.pay, 63_500)

	def test_monthly_schedule(self):
		'''here we use patch a context manager for mocking
		mocking allows us to still test our code even if an external variable, like the 
		operation of an external website is down without our test returning a Failed result
		because of an external event
		
		What we pass to patch is what we want to mock, is requests.get from the employee module, and setting
		that equal to mocked_get. We didn't just import it straight out but we want to 
		mock these objects where they are actually being used in the script.
		'''
		with patch("employee.requests.get") as mocked_get:
			# Testing a passing value
			mocked_get.return_value.ok = True
			mocked_get.return_value.text = "Success"

			# within our context, we want to run our method monthly_schedule method
			# just like we are testing it.
			schedule = self.emp_1.monthly_schedule("May")
			mocked_get.assert_called_with("http://company.com/Schafer/May")
			self.assertEqual(schedule, "Success")

			# Testing a failed response
			mocked_get.return_value.ok = False

			# within our context, we want to run our method monthly_schedule method
			# just like we are testing it.
			schedule = self.emp_2.monthly_schedule("June")
			mocked_get.assert_called_with("http://company.com/Smith/May")
			self.assertEqual(schedule, "Bad Response!")
>>>>>>> 127c6c57686d95ac80fb811aa750727e3a136371

'''
as programmers, we often try to keep our code DRY
DRY stands for: Don't Repeat Yourself
We we see a lot of similar classes that appear and this means that if we make updates
it will be a pain to just have to fix those things in all those spots

So why not put all the testcases in one place and re-use them for every test?

That is what the setup() and teardown() methods are for.
<<<<<<< HEAD
You can do this by creating two new methods at the top of our TestClass.

The print statements indicate that the tests are run as such:
-setUp
-test_
-tearDown

We also notice that the tests are not necessarily run in order of how they appear in the script (i.e., from top to down)
That's why we need to keep our code isolated from one another.
And that's why it's useful to have some code run at the very beginning of test file,
and then have some cleanup code that runs after all the tests have been run.

So, unlike the setUp and tearDown that runs before and after every single test, it would be nice
if we had something that would run once before for anything and once after everything.
We can actually do this with two class methods called setUPClass and tearDownClass
Remember, you can create class methods with the @classmethod decorator above the method
Classmethods allow us to work with the class itself, rather than the instance of the class.

Notice how setUpClass and tearDownClass are run at the start off all the testing
and tearDownClass is run after all the testing is done
setUpClass and tearDownClass are useful if you want to run something once and is too costly to do before each test. 
=======

Also, last notes:
Test should be isolated. Basically this just means that your tests should run without affecting other tests.
In this video, Corey was adding tests to existing code. 

You might have heard of something called 
"Test Driven Development". Basically what test-driven development means is that you write the test BEFORE
you even write the code. Sometimes this can be useful. This is not always followed in practice but it is nice.
The concept is: you should think about what you want your code to do and write a test implementing that behaviour 
and then watch the test fail since it doesn't have any code to run against and then to write the code in a way
that gets the code to pass.

Simple tests and testing is better than no testing. Don't feel like you have to be an expert at writing mocks 
and things like that. Even if you just write some basic assertions, then it's better than just not writing anything.
There is also another test framework out there called PyTest than a lot of people like to use than this built-in unittest module. 
>>>>>>> 127c6c57686d95ac80fb811aa750727e3a136371

'''
	# define a new test for testing the new employee method
	def test_monthly_schedule(self):
		


if __name__ == "__main__":
	unittest.main()