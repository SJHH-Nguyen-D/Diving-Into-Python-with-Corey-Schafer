"""
decorate modifies the behaviour of the wrapped function without permanently changing the functionality. it can copy the function signature of the old function for easy access, as well as keep the metadata from the wrapped function using the functools.wraps function decorator
"""

import functools 
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)

def trace(func):
    """ Decorate a closure to change """
    @functools.wraps(func)
    def decorated_function(*args, **kwargs):
        print(func, args, kwargs)
        result = func(*args, **kwargs)
        pp.pprint(result*kwargs["times"])
    return decorated_function

@trace
def greet(greeting, name, *args, **kwargs):
    """ Prints a greeting to a person """
    kwargs['case'] == 'default'
    if kwargs['case']=='lower':
        greeting, name = greeting.lower(), name.lower()
    elif kwargs['case']=='upper':
        greeting, name = greeting.upper(), name.upper()
    else:
        greeting, name = greeting.capitalize(), name.capitalize()
    return f"{greeting}, {name}! arrrrrggghhhs: {args}, kwargggs:{kwargs}"

greet("Hello", "Bob", "Peter", 2.0, np.random.randint(1, 10), case='upper', times=3)
print(f"This is the trace docstring: {trace.__doc__}")
print(f"This is the greet docstring: {greet.__doc__}")

def print_vec(*args):
    print(f"\n<{', '.join(args)}>\n")

list_vec = ["alpha", "kilo", "bezos"]
tuple_vec = ("alpha", "kilo", "bezos")

""" using a * operator on a sequential iterable like a string, list, tuple or generator in passing it as an argument to a function lets you unpack it. Using the ** operator, you can unpack dictionary values passed as an argument which is handy for things like kwargs, but you can also just unpack the keys with just the * operator"""
print_vec(*list_vec)
print_vec(*tuple_vec)