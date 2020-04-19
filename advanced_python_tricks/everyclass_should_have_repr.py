"""
REPR and STR to string conversion of representations of your class objects.

__str__ is the human readable textual representation of your object that you want to expose to your non-developer users.

__repr__ is the developer string representation that you want to show your developers that is more explicit as it usually includes information about the class such as the module/submodule, and arguments passed to it.
"""

import datetime

class Car:
    def __init__(self, color, mileage, *args, **kwargs):
        self.color = color
        self.mileage = mileage
        self.args = args
        self.kwargs = kwargs
    
    def __str__(self):
        return f"a {self.color} car"

    def __repr__(self):
        """
        note the !r suffix, indicating that we want the repr value supplied (i.e., repr(value)). Use !s for the str(value) of a value or !a suffix for the ascii value.
        """
        return f"{self.__class__.__name__}({self.color!r}, "\
               f"{self.mileage!r}, args={self.args}, kwargs={self.kwargs})"

mycar = Car('blue', 1000, 'asdlfjasdf')
print(f"repr of car: {repr(mycar)}")
print(f"str of car: {str(mycar)}")

today = datetime.date.today()

print(f"Repr of datetime object: {repr(today)}")
print(f"String representation of the datetime object{str(today)}")
print(f"type of today: {type(today)}")
print(f"type of today: {type(str(today))}")