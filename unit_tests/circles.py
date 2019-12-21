from math import pi

def circle_area(radius):
	''' Calculates the area of a circle given a radius '''
	
	# Ensure that the radius is an int or a float
	if type(radius) not in [int, float]:
		raise TypeError("The radius be a non-negative real number.")

	# Ensure that the radius is positive
	if radius < 0:
		raise ValueError("The radius cannot be negative.")
	return pi*(radius**2)



# Test function
'''remember, in python, to create complex numbers, which include the use
of imaginary numbers, the character j is equivalent to sqrt(-1)
'''
# radii = [2, 0, -3, 2+5j, True, "radius"]

# for radius in radii:
# 	message = "Area of a circle with r = {} is {}".format(radius, circle_area(radius))
# 	print(message)