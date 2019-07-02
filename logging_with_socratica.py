'''
Logging

Video follow along with socratica.

Logging allows us to write messages to a file or other outputs stream. These messages contain information on which parts of your code have executed and what problems may have arisen.

Each of the messages has a level
1. Debug
2. Info
3. Warning
4. Error
5. Critical

If you want, you can create additional levels but these should be sufficient. 

Link to the video:
https://www.youtube.com/watch?v=g8nQ90Hk328
'''
import logging, os
import math

# All Caps = CONSTANTS
# Capitalized = Classes
# camelCase = methods
print(dir(logging))

# keep it simple with a basicConfig logging method
# create a and configure a logger with the .getLogger() method
filename = os.path.join(os.getcwd(), "lumberjack.log")

# To change the format of the logging information, create a string of the logging attributes
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename=filename, 
	level=logging.DEBUG, 
	format=LOG_FORMAT,
	filemode="w")

# you can give your logger a name if you are working with a lot of loggers
# if you don't this is the main logger and it is known as the root logger
logger = logging.getLogger()

# test the logger at the INFO level
logger.info("Our second message.")

# test the logger at the DEBUG level
logger.debug("This is a harmless debug message.")

# test the logger at the WARNING level
logger.warning("This is a warning message.")

# test the logger at the ERROR level
logger.error("Some things are going terribly wrong!")

# test the logger at the CRITICAL level
logger.critical("This is a critical warning")

print(logger.level) # prints out 30
# in the logger module, there are 6 builtin level constants:
# NOTSET = 0
# DEBUG = 10
# INFO = 20
# WARNING = 30
# ERROR = 40
# CRITICAL = 50

'''
loggers will only write a message equal to or greater than the set level
our test log message was an info message with a value of 20.
The basicConfig sets the level of the root logger to 30, which is the WARNING level, by default.
you change the level of the basicConfig call to the level of DEBUG above.

Notice that the log message contains a message of the following signature:
LEVEL: the logger : The message

This format is okay, but it is missing an important piece of informmation - the time the log message was created.
Let's now change the format of the log message. Python includes a wide selection of information
you can include within your log messages. You can view theses attributes on the www.python.org website.

In this video we are mainly interested in the level, asctime, and message attributes of the logger

You may also notice that by default, the logger is in append mode, meaning that 
the previous writes to the logging file will still be there and additional entries
will simply be appended at a new line in the log.

You can change it to overwrite mode by changing the filemode="w".

You can log messages of any level by calling the level name. 
'''

# We will now try to apply logging in a real scenario

def quadratic_formula(a, b, c):
	""" Return the solutions to the equation ax^2 + bx + c = 0. """
	logger.info("quadratic_formula({0}, {1}, {2})".format(a, b, c))

	# compute the disciminant
	# the discriminant is the number under the sqrt
	# before you write the formula, now would be a good time for another debug message
	logger.debug("# Compute the discriminant") # once we are sure that this is working as planned, we may want to remove the debug logs and keeps the comments
	discriminant = b**2 - (4*a*c)

	# compute the two roots
	logger.debug("# Compute the two roots")
	root1 = (-b + math.sqrt(discriminant)) / (2*a)
	root2 = (-b - math.sqrt(discriminant)) / (2*a)

	# return the two roots as a tuple
	logger.debug("# Return the roots")
	return (root1, root2)

roots = quadratic_formula(1, 0, 1) # returns an error about the origin
print(roots)

