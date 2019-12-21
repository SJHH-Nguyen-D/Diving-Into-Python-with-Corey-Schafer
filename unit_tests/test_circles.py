'''
Similar to the video that Corey Schafer did on unittesting 
with the unittest module in python, Socratica made a video of the same topic.
This is the testing follow along script that I followed along with for the video.
This video is also to reinforced what I just learned from Corey Schafer's unittesting video

The link to the Socratica's YouTube video is here: https://www.youtube.com/watch?v=1Lfv5tUGsn8
'''

from circles import circle_area
from math import pi
import unittest
from unittest.mock import patch

# As always, create a class that is a subclass of the TestCase class from the unittest module
class TestCircleArea(unittest.TestCase):
	''' Test class for testing circle area calculator '''

	# __init__ called everytime a new object is created. self is a reference to the object created)
	def __init__(self, )

	@classmethod
	def setUpClass(cls):
		print("setUpClass\n")

	@classmethod
	def tearDownClass(cls):
		print("tearDownClass")

	def setUp(self):
		print("setUp")
		# self.radii = [2, 0, -3, 2+5j, True, "radius"]

	def tearDown(self):
		print("tearDown\n")

	def test_circle_area(self):
		print("test_circle_area")

		# Test areas when radius >=0
		self.assertAlmostEqual(circle_area(1), pi) # to 7 decimal places
		self.assertAlmostEqual(circle_area(0), 0)
		self.assertAlmostEqual(circle_area(2.1), pi*2.1**2)

	def test_value(self):
		print("test_value")
		# make sure value errors are raised when necessary
		with self.assertRaises(ValueError):
			circle_area(-2)

	def test_type(self):
		print("test_type")
		# make sure that a type error is raised when the radius is not a real number or when
		# or when necessary

		# Raise a TypeError for a Complex Number
		with self.assertRaises(TypeError):
			circle_area(3+5j)

		# Raise a TypeError for a Boolean
		with self.assertRaises(TypeError):
			circle_area(True) # notice that a TypeError was note raised for this when first run
							  # this means that we have to adjust our function to raise an error for booleans

		# Raise a TypeError for a String
		with self.assertRaises(TypeError):
			circle_area("radius")

if __name__ == "__main__":
	unittest.main(exit=False)

'''
In the terminal, if you haven't set unittest as the main script,
You'd have to enter this in the terminal when you run it:

python -m unittest test_circles

The -m option tells python to run the unittest module as the script.
'''