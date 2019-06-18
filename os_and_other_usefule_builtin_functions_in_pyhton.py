# This script goes thhrough various useful functions from the os module as well as other useful built-in libraries
import os
filename = os.path.join(os.environ.get("HOME"), "test.txt") # example filename that we will use for this walkthrough

# listing all the files and folders inside the specified directory
print(os.listdir())


################################# FILE INFORMATION WITH OS.STAT #########################################
# This method allows us to see the statistics of a particular file
print(os.stat('notesfromcharlie_session.md'))

# say if you want to know the modification time of a file you could pass in this bit from the file: # st_mtime=1560099666
print(os.stat('notesfromcharlie_session.md').st_mtime) # you get something that isn't human readable because it is in linux time


############ CONVERTING EPOCH TIME TO READABLE TIME WITH DATETIME.DATETIME.FROMTIMESTAMP ##########
# so in order for us to understand what this means, you have to import a function from the datetime module to do that
from datetime import datetime

modification_time = os.stat('notesfromcharlie_session.md').st_mtime
print(datetime.fromtimestamp(modification_time))


########################## DIRECTORY WALK WITH OS.WALK()##################################################
# prints a bunch of tuples which contains the file tree of the specified directory.
# In this case, the specified directory is the current working directory
for i in os.walk(os.getcwd()):
	print(i)

print(len(list(os.walk(os.getcwd())))) # more than a hundred files in this directory tree 

# you can also unpack that generator as each element is a tuple with the following tuple structure
# you can look through this entire walk of files in the directory to see if a file that you are looking for exists
# This is actually a very useful function that a lot of people tend to use.
# Maybe the one tricky part is remembering how the arguments for this os.walk() function are unpacked
# The unpacking signature is as follows: dirname, dirpath, filename which corresponds to full pathname, the directories in that path, and the files in that directory

for dirpath, dirnames, filenames in os.walk(os.getcwd()):
	print("current path: {}".format(dirpath))
	print("directories: {}".format(dirnames))
	print("Files: {}".format(filenames))
	print() # this is to have a space in between each file


############################# ENVIRONMENT VARIABLES WITH OS.ENVIRON.GET() ##########################################
# Say that you want to access your HOME path information
# you can get your environment variables by using environ.get() method as such:
# The output is my home directory
print(os.environ.get("HOME")) # returns the directory where the HOME environment is pointing to in the environment variables

################################# GET THE FILENAME/LAST DIR NAME OS.PATH.BASENAME #################################
# if you want the file name of just the file itself use the os.path.basename() method to get the file name
print(os.path.basename(filename)) # returns the filename with the extension attached

#################################### DOES A PATH EXISTS: OS.PATH.EXISTS ############################################
# If you want to check if a path or file exists, you can use the os.path.exists() method that checks to see if it exists
print(os.path.exists(filename)) # returns a boolean

#################################### DOES A FILE EXISTS: OS.PATH.ISFILE ############################################
# If you wanna check if a thing is a file or not and if it exists, you can use the os.path.isfile() method
print(os.path.isfile(filename)) # returns a boolean

#################################### DOES A DIR EXISTS: OS.PATH.ISDIR ############################################
# Like the os.path.isfile() method, you can use the os.path.isdir() method to see if a thing is a directory or not
print(os.path.isdir(filename)) # returns a boolean

#################################### DOES A PATH EXISTS: OS.PATH.SPLIT() ############################################
# if you want to have split the path name into their separate directories as strings, you can use the split method
# It returns a list of strings of directories
print(os.path.split(filename)) # returns a tuple of strings of the full path as the first element and the filename as the second element

#################################### EXTRACT THE FILE EXTENSION AS THE LAST ELEMENT: OS.PATH.SPLITEXT() ############################################
# If you want to split out the file extension from the full path to file name, you can use teh splitext() method
# This is a very useful function that you haven't used a lot yet and maybe you'll find yourself using it a lot in the future
print(os.path.splitext(filename)) # returns a tuple of strings with the first element being the full path with the filename, and the second being just the file extension
