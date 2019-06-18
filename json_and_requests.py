import os
import json
import requests

"""
In this script, we will dive into grabbing json files from the internet with the request package 
and we will also use the JSONPlaceholder, a great source of fake JSON data for practice purposes
"""
response = requests.get("https://jsonplaceholder.typicode.com/todos")
todos = json.loads(response.text)

todos == response.json()

type(todos)

todos[:10]

# Map of userId to number of complete TODOs for that user
todos_by_user = {}

# Increment complete TODOs count for each user.
for todo in todos:
    if todo["completed"]:
        try:
            # Increment the existing user's count.
            todos_by_user[todo["userId"]] += 1
        except KeyError:
            # This user has not been seen. Set their count to 1.
            todos_by_user[todo["userId"]] = 1

# Create a sorted list of (userId, num_complete) pairs.
top_users = sorted(todos_by_user.items(), 
                   key=lambda x: x[1], reverse=True)

# Get the maximum number of complete TODOs.
max_complete = top_users[0][1]

# Create a list of all users who have completed
# the maximum number of TODOs.
users = []
for user, num_complete in top_users:
    if num_complete < max_complete:
        break
    users.append(str(user))

max_users = " and ".join(users)

s = "s" if len(users) > 1 else ""
print("user{} {} completed {} TODOs".format(s, max_users, max_complete))

# Define a function to filter out completed TODOs 
# of users with max completed TODOS.
def keep(todo):
    is_complete = todo["completed"]
    has_max_count = str(todo["userId"]) in users
    return is_complete and has_max_count

# Write filtered TODOs to file.
with open("filtered_data_file.json", "w") as data_file:
    filtered_todos = list(filter(keep, todos))
    json.dump(filtered_todos, data_file, indent=4)


# This is what happens when you work with Classes and try to serialize a class from 
# python as a json. You will have to do  a bit fo work first before you get it to work

class Elf:
	def __init__(self, level, ability_scores=None):
		self.level = level
		sefl.ability_scores = {
		"str": 11, "dex": 12, "con": 10,
		"int": 16, "wis": 14, "cha": 13
		} if ability_scores is None else ability_scores
		self.hp = 10 + self.ability_scores["con"]

'''
Although the json module can handle most buil-in python types, it doesn't
understand how to encode customized data types by defailt. It's like trying to fit a 
square peg into a round hole
'''

'''
Now, the question is how to deal with more complex data structures. Well, you could try to 
encode and ecode teh JSON by hand, but there's a slightly more clever solution
that'll save you some work. Instead of going straight from the custom data type 
to JSON, you can throw in an intermediary step.

All you need to do is represent your data in terms of the built-in types json already understands.
Essentially, you translate the more complex object into a simpler
representation, which the json module then translates into JSON.
It's like the transitive property of mathematics: if A=B then B=C and A=C.

To get the hang of thism, you'll need a complex object to play with.
You could use any custom class you like, but Python ahs a built-in type called
complex for representing complex numbers, and it isn't serializable by default. So, for the sake of these examples,
your complex object is going to be a complex object. Confused yet?
'''

# below, we get that this code is not serializable
z = 3 + 8j
# print(type(z))

# print(json.dumps(z))
'''
A good question to ask yourself when working with custom types is: What
is the minimum amount of intormation necessary to recreate this object? In the case of
complex numbers, you only need to know the real and imaginary parts, both of which you can accesss as attributes
on the complex object
'''
print("This is real number of z: {}".format(z.real))
print("This is the imaginary number of z: {}".format(z.imag))


'''
Passing the same numbers into a complex constructor is enough to satisfy the __eq__ comparison operator:
'''
print("Z is a complex number? : {}".format(complex(3, 8) == z))

'''
Breaking custom data types into their essential components is critical to both the serialization and deserialization process
'''

'''Encoding Custom Types

To translate a custom object into JSON, all you need to do is provide an encoding
function to the dump() method *default* parameter. The json module will call this function on
any objects that aren'y natively serializable. Here's a simmple decoding fucntion
you can use for practice:
'''

def encode_complex(z):
	if isinstance(z, complex):
		return (z.real, z.imag)
	else:
		type_name = z.__class_name__
		raise TypeError("Object of type {} is not JSON serializable".format(type_name))

print(json.dumps(9 + 5j, default=encode_complex))
# The below-line is still not serializable as  JSON file
# print(json.dumps(elf, default=encode_complex))

'''
The other common apporach is to subclass the standard JSONEncoder and override its default() method:
'''

class ComplexEncoder(json.JSONEncoder):
	def default(self, z):
		if isinstance(z, complex):
			return(z.real, z.imag)
		else:
			return super().default(z)

print("The output of subclassing from the standard JSONEncoder in the json module: {}".format(json.dumps(2, 5j, cls=ComplexEncoder)))

encoder = ComplexEncoder()

print("An example of our custom encoder: {}".format(encoder.default(3 + 6j)))

'''
Decoding Custom Types

While the real and imaginary parts of a complex number are absolutely necessary,
they are actually not quite sufficient to recreate the object.
This is what happens when you try encoding a complex number with the ComplexEncoder
and then decoding the result:
'''

complex_json = json.dumps(4 + 17j, cls=ComplexEncoder)
print(json.loads(complex_json))

'''
All you get back is a list, and you'd ahve to pass the values into a complex constructor
if you wanted that complex object again.
Recall our discussion about teleportation. What's missing is metadata, or information about the type of data
you're encoding

I suppose the question you really ought ask yourself is What si the minimum amount of information
is bouth necessary and sufficient to recreate this object?

The json module expects all custom types to be expressed as objects 
in the JSON standard. For variety, you can create a json file this time called complex_data.json
and add the following object representing a complex number
'''

complex_json = {
    "__complex__": True,
    "real": 42,
    "imag": 36 }

with open("complex_data.json", "w") as write_file:
	json.dump(complex_json, write_file)

'''
The complex key is the metadata that we talked about.
It doesn't really matter what the associated value is. To get this little hack to work, all you need to do is
verify that the exists:
'''

def decode_complex(dct):
	if "__complex__" in dct:
		return complex(dct["real"], dct["imag"])
	return dct

'''
If ___complex___ isn't in the dictionary, you can just return the object and let the default decoder deal with it.

Every time the load() method attempts to parse an object, you are given the opportunity to intercede before
the default decoder has its way with the data. You cacn do this by passing your decoding function to the object_hook parameter
'''

with open("complex_data.json") as complex_data:
	data = complex_data.read()
	z = json.loads(data, object_hook=decode_complex)

print(type(z))
print(os.listdir("./"))

"""
While the object_hook might feel like the counterpart to the dump() method's default parameter,
the analogy really begins and ends there.

This doesn't just work with one object either. Try putting this list of ocmplex numbers 
into complex_data.json and running the script again:
"""

complex_data = [
  {
    "__complex__":True,
    "real":42,
    "imag":36
  },
  {
    "__complex__":True,
    "real":64,
    "imag":11
  }
]

with open("complex_data.json", "w") as write_file:
	json.dump(complex_data, write_file)

'''
if all goes well, you'll get a list of complex objects:
'''

with open("complex_data.json") as complex_data:
	data = complex_data.read()
	numbers = json.loads(data, object_hook=decode_complex)

print(numbers)

"""
congrats. you can now wield the mighty power of json for any and all 
of your python needs.

while the examples you've worked with here are certainly contrived and overly simplistic,
they illustrate a workflow you cacn apply to more general tasks:
1. Import json package
2. read the data with load() or loads()
3. processe the data
4. write the altered data with dump() or dumps()

Today you took a journey: you captured and tamed some wild JSON, and you made it back
for supper! As an added bonus, learning the json package will make learning
pickle and marshal a snap.

"""



