import subprocess
import os

"""
There are many cases in which you want to call an external command 
in python and you can do this with the built-in library called subprocess.
And if you need to, you can also capture the output of that command or even pipe the output 
from one command into another. 
"""

# running an external command is pretty simple with the subprocess module
# in this case, we run the command line command "ls"
subprocess.run("pwd")
print(type(subprocess.run("pwd"))) # returns a subprocess.CompletedProcess class
print(subprocess.run("pwd"))

print(os.getcwd())
print(type(os.getcwd())) # returns a string

subprocess.run("ls -alt", shell=True) # you can use shell=True
# one downside to using shell=True is that it can be a security hazard, as you are passing in the arguments yourself
# So only use this if you are passing in the arguments yourself, and be sure you're not running it with user input or anything like that.

# but, if we don't want to specify using shell=True, you will need to pass in everything as a list of arguments and we get the same thing
subprocess.run(["ls", "-alt"])

# we can get the arguments that were passed in as part of the original command with the .args function
a = subprocess.run(["ls", "-alt"], capture_output=True)
print(a.args)
# we can also see if we get any errors from the function by looking at the return code. 0 means exited without any issues.
print(a.returncode)

# you can also grab the standard output of the function
# the reason we are getting back none as a result is that is is just sending the standard out to the console
print(a.stdout) # we get back None as a result
print(a.stdout.decode())

# if we want to capture the output of this function, we will have to pass into the function, a special argument to indicate
# that we want to capture the output of this function
# z = subprocess.run(["ls", "-alt"], stdout="PIPE") # for some reason, I am getting expected argument error
# print(z)
