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
	
	def monthly_schedule(self, month):
		''' Goes to a website to fetch a monthly schedule for the employee '''
		response = requests.get(f"https://company.com/{self.last}/{month}")
		
		try:
			response.raise_for_status()
		except Exception as exc:
			print("There was a problem: {}".format(exc))

		if response.ok:
			return response.text
		else:
			return "Bad Response!"