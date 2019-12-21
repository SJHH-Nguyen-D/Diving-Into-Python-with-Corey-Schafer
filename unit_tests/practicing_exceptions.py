'''
In this script, we practice writing exceptions.

Exceptions or exception objects are what are raised when your program encounters an error.
Exceptions contain:
- a description of what went wrong
- and a traceback of where the error occured in the script

Typically, writing exceptions come in the form of this:

try:
	# Runs code in this block first
	# If no problem occurs, after the try block, python will skip all the except blocks
	# and RUN the code in the ELSE block and then RUn the FINALLY block
	# <code>
	pass
except:
	# If an error occurs, jump to this except block
	# <code>
	pass
except: 
	# you can have more than one exception and different types of exceptions
	# <code>
	pass
else:
	# The code in the ELSE block runs if the try block code runs successfully
	# <code>
	pass
finally:
	# The code in the FINALLY block runs regardless of what happens above, or no error
	# <code>
	pass

You can use one of the exceptions from the built-in exceptions or
you can make your own by subclassing from the builtin exception class.

'''

'''
EXAMPLE

Objective:
- write a binary file and return the data
- measure the time required to do this
'''
import timeit
import logging # for logging the results of our run
import os

# First create a logger with basic debug level
logging.basicConfig(filename=os.path.join(os.getcwd(), "problems.log"),
	level=logging.DEBUG)
logger = logging.getLogger()

def read_file_timed(path):
	''' return the contents of the file at path and measure time required. '''
	start_time = timeit.default_timer()
	try:
		with open(path, "rb") as read_file:
			data = read_file.read()
			return data
	# name the FileNotFoundError object err
	except FileNotFoundError as err: 
		# we log the error
		logger.error(err)
		# we type the raise command to tell Python
		# to we pass along the FileNotFoundError to the user
		raise
	else:
		# This code only executes if there are no exceptions from the try block
		pass
	finally:
		stop_time = timeit.default_timer()
		dt = stop_time - start_time
		logger.info("Time required for {} = {}".format(os.path.filename(path), dt))

path = '/home/dennis/Desktop/Link to datascience_job_portfolio/notesfromcharliesession.md'
data = read_file_timed(path)