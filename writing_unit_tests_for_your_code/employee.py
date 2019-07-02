import requests

class Employee():	
	""" A sample Employee class """

	raise_amt = 1.05

	def __init__(self, first, last, pay):
		self.first = first
		self.last = last
		self.pay = pay


	@property
	def email(self):
		''' Create email given first name and secondname'''
		return "{}.{}@email.com".format(self.first, self.last)


	@property
	def fullname(self):
		''' Create fullname with first and last name '''
		return "{} {}".format(self.first, self.last)


	def apply_raise(self):
		''' Update the pay amount by a percentage '''
		self.pay = int(self.pay * raise_amt)


'''
There is another thing about unit testing that is important for most people to know.
Sometimes our code relies on certain things that we have no control over.
For example, let's say we have a function that goes to a website, and pulls down some information./
Now, if that website is down, and your function is going to fail, which will also make your tests fail. 
But this isn't what we want, because we only want our test to fail if something is wrong with OUR code. 
So if a website is down, then there is nothing we can actually do about that. 
So we are going to have to get around this with something called "Mocking".

There's a lot you can do with Mocking, but let's take an example of some basic usage.

The function below uses the employee's last name to create a web address that accesses their monthly schedule information
So, the information from that website is something that we would want to Mock because we don't want the success of our test to be
dependent on that website being up. 

We only care that the get method was called with the correct URL and that our code behaves corrctly whether the response was okay 
and whether the response was not okay. We can do this by importing a method from unittest.mock called patch, at the top.
'''
	def monthly_schedule(self, month):
		response = requests.get(f"http://company.com/{self.last}/{month}")

		try:
			response.raise_for_status()
		except Exception as exc:
			print("There was a problem: \n{}".format(exc))

		if response.ok:
			return response.text
		else:
			return "Bad Response!"


		
