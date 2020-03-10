'''
Video follow along with corey schafer of the same title.
Video can be found here: https://www.youtube.com/watch?v=C-gEQdGVXbk
'''

################### Ternary Operators or Ternary Conditionals ###################
# Can very very useful and save you sometime from writing a lot of unnecessary code
# This is the regular way would might have typed it

condition = False
if condition:
	x = 1
else:
	x = 0
print(x)

# But you can write it more concise in this manner
# The format of this goes: 
# variable = value if condition else vallue
x = 1 if condition else 0
print("The more concise way of writing the above code in one line: {}".format(x))

# Just because it is a one liner doesn't mean you are writing better code
# The point of the code is that it should be cleaner but also easier to read for you and your viewers

####################### Large Numbers #############################
num1 = int(10000000000)
num2 = int(10000000000000)
total = num1 + num2
print(total)

# The question is, how many digits of 0 are there in each of these, just reading from a glance?
# We can increase the legibility of reading how many digits of 0 there are by adding separators
# Instead of adding commas in real life to separate the zeros, you can use underscores (_) instead and it wouldn't affect reading your code

num1 = int(10_000_000_000)
num2 = int(10_000_000_000_000)
total = num1 + num2
print(total)

# Further increase legibility with f-string, curly braces and colon for formatting
# After the colon, you want to pass in whatever you want to use as the separator. Has to be a recognised separator (, or _)
print(f'{total:,}')
print(f'{total:_}')

##################### Opening and closing a file #########################

# This is the typical way that one would tackle closing and opening a file
# This is manuall managing the resources of your computer for the file that is being opened
# This should off alarms inside your head and this is something that people call a "code-smell" because it doesn't look or smell right
f = open("test.txt", "r")
file_contents = f.read()
f.close()

# This is an alterative way to open and close a file that is much safer in memory, with the context manager
with open("test.txt", "r") as f:
	file_contents = f.read()

words = file_contents.split(" ")
words_count = len(words)
print(words_count)
# If you are working with threads and are manually acquiring and releasing locks, you can use context manageres to do that for you. 
# When you are working with resources, the point is to train yourself to notice when you need to use a context manager pass in when you need your
# resources to be handled automatically instead of doing it manaually 

################################# The Enumerate Method #######################

# If you have to keep track of the index position of the what the item is in the list, you might do something like this to get it
names = ['Corey', 'Daniel', 'Nathan', 'Matthew', 'Carmichael']
for n in names:
	print('Position number: {},\nName: {}\n'.format(names.index(n), n))

# But instead of doing this, you can use the ENUMERATE function to automatically handle this for you if you are going through a list
# As it will automatically pack in the index position number as well as the item in the list
names = ['Corey', 'Daniel', 'Nathan', 'Matthew', 'Carmichael']
for n in enumerate(names):
	print('Position number with enumerate: {},\nName with enumerate: {}\n'.format(n[0], n[1]))

# You can also unpack it like this
names = ['Corey', 'Daniel', 'Nathan', 'Matthew', 'Carmichael']
for index, name in enumerate(names, start=1):
	print('Position number with enumerate, unpacked: {},\nName with enumerate, unpacked: {}\n'.format(index, name))

# You may also be writing bad code because you are not aware that a function exists.
# say you want to loop over two lists at once
names = ['Corey', 'Daniel', 'Nathan', 'Matthew', 'Carmichael']
code_names = ['Burger', 'Squirter', 'Black', 'Lemon', 'El Cucaracha']
universes = ['Planet 7', 'MCU', 'DC', 'MCU', 'Planet 7']

# This is how it has been done, which is 'incorrect' per se and there is a better way for people to do something like this
# You might think to use the index of the code name in the other list which corresponds to the real name in the other list
# using the enumerate function we found in the previous example
for index, name in enumerate(names):
	code = code_names[index]
	print(f'{name} is actually {code}')

# The way that we should be doing something like this in python is to use the zip function, which we have seen in the use of making dictionary comprehensions
# It allows us to iterate through two lists at once and both values of the list at once. You can even use zip for more than two lists at once
# This is actually more clean and intuitive than before and it also erases the need for a placeholder variable line
# f-strings are when you place an f in front of string and anything within the string with a curly brace, you are formatting with a placeholder
for name, code_name, universe in zip(names, code_names, universes):
	print(f'Using the zip function, {name} is actually {code_name} from {universe}')

# If you are using lists of different lengths, zip will automatically stop at the index of the shortest list
# However, if you want to have it loop through the whole thing, regardless of the shortest-list stipulation, you can import the zip method from the built-in itertools module
# from itertools import ZipLongest

# if you want a tuple of all three of those joint values you can print out it as a single parsed item
for record in zip(names, code_names, universes):
	print(record)

########################## Unpacking Values #############################################

# This does whatever you think it does
a, b, c, d, e = (1, 2, 3, 4, 5)
print(a, b, c, d, e)

# If you want to unpack just the first two values and ignore the rest
a, b , *_ = (1, 2, 3, 4, 5)
print(a, b)

# you can also achieve the same thing but keep the assignment of some of the attributes later down the line
a, b, *_, d = (1, 2, 3, 4, 5)
print(a, b, d)

######################## Setting and Getting Attributes with setattr() and getattr() when working with class objects######################

class Person():
	def __init__(self, value):
		self.value = value

person = Person(value=5)

# you can set attributes of a class and create attributes of a class and pass in its values as such
person.skill = "thievery"
print(person.skill) # you've set a new attribute here

# However if you want to set an attribute of an instance or class to a variable, you might run into an error
intelligence = 10
person.intelligence = intelligence
print(person.intelligence) # but you don't in this case

# You should actually go about this as with the setattr() method.
# the signature is as follows setattr(instance/object, attribute, value)
setattr(person, "first_name", "Daniel")
print(person.first_name)

# However, unlike before, we CAN set attribute values using variables
my_address_attribute = "address"
address = "564 Earth Street, Omaha, Nebraska"
setattr(person, my_address_attribute, address)
print(person.address)

# You can also retrieve the attribute value of an object with the getattr() function
# getattr has the signature: getattr(object, attribute)
print(getattr(person, my_address_attribute))\

# these functions make it super easy to then work with looping over attributes/keys in a dictionary to perform some function
# Going through the loop can be done as such
person_info = {"first": "Corey", "last": "Schafer"}

# the dict.items() method allows us to get the key and the value when working with a dictionary
for key, value in person_info.items():
	setattr(person, key, value)

# we've set the attributes into this list as such and now we can see with this 
print(person.first, person.last)

# If you wanted to print those attributes in a list as will you can also do something like this
for key in person_info.keys():
	print(getattr(person, key))
	print(key)


############################ Inputting Secret Information ##################################
# python has a function called getpass() which is typically used for when your script requires an input from the user to run
# First let's look at the wrong way to get an input from the user. 
# Keep in mind that, using a script that requires the input of a user from sublime usually doesn't work out too well.

# username = input("Username: ")
# password = input("Password: ")
# print("Logging in ...")
# print(f"Username: {username}, Password: {password}")

# The above script unfortunately displays the user's information upon entry and this is not a feature that users typically want
# luckily, python has a getpass() function from the getpass built-in module
# You use it in the same manner as using the built-in input() function
from getpass import getpass

username = input("Username: ")
password = getpass("Password: ")

print(f"Username: {username}, Password: {password}")


########################## Running Python with the dash M option ##############################
# We typically run a script from the command line with this command:
# python myscriptname.py

# The reason why we are able to do this is because that script is in our current directory
# If you want to run a script not in our current directory, you have to use the python -m option as such
# NOTE: that you do not want to add the .py to the end of that file because you are just specifying the module name
# the reason why this works is because when you use the -m option is that it is going to search our sys.path for that module and our current directory is added to our sys.path
# python -m myscriptnamefromanotherdirectory

# How do you know what options you can run that module with from the command line?
# Because you ran that module with the -m option, that also means you can import that module as well and that means you can import anything from your sys.path
# That means you can import that module and have a further look at it. 

######################### Help and Dir function #################################################
# you've seen this before
# you can use the help() and dir() built-in functions on any package or object to get the information on which methods and attributes the stuff has
from datetime import datetime

for i in dir(datetime):
	print(i)

print(datetime.today()) # learned that this
print(datetime.today) # we learned from this that datetime.today is a METHOD and NOT an attribute because of the python help tip


######################### Lambda Functions #######################################################
# Lambda functions are anonymous functions that don't need a name because they are relatively shortlived and don't need a name
# They use the lambda keyword to start and they take on the signature as follows:
# lambda input_variable_name: output_variable_expression_thatuses_that_input_variable_on_the_lefthand_side
squareit = lambda x: x**2
print(squareit(10))

