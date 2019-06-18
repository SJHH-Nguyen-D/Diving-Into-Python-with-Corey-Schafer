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
	
