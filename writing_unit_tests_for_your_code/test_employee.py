import unittest
from employee import Employee

class TestEmployee(unittest.TestCase):

	def test_email(self):
		# we create example objects so that we can start the testing
		emp_1 = Employee("Corey", "Schafer", 50_000)
		emp_2 = Employee("Sue", "Smith", 60_000)

		# we test if those assertions on those new instances hold true
		self.assertEqual(emp_1.email, "Corey.Schafer@email.com")
		self.assertEqual(emp_2.email, "Sue.Smith@email.com")

		# we change the values for the names
		emp_1.first = "John"
		emp_2.first = "Jane"

		# we check if given the assertions that the values still hold true
		self.assertEqual(emp_1.email, "John.Schafer@email.com")
		self.assertEqual(emp_2.email, "Jane.Smith@email.com")

	def test_apply_raise(self):
		# we create example objects so that we can start the testing
		emp_1 = Employee("Corey", "Schafer", 50_000)
		emp_2 = Employee("Sue", "Smith", 60_000)

		emp_1.apply_raise()
		emp_2.apply_raise()

		self.assertEqual(emp_1.pay, 52_500)
		self.assertEqual(emp_2.pay, 63_500)

'''
as programmers, we often try to keep our code DRY
DRY stands for: Don't Repeat Yourself
We we see a lot of similar classes that appear and this means that if we make updates
it will be a pain to just have to fix those things in all those spots

So why not put all the testcases in one place and re-use them for every test?

That is what the setup() and teardown() methods are for
'''

if __name__ == "__main__":
	unittest.main()