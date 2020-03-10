import json
import os

data = {
    "president": {
        "name": "Zaphod Beeblebrox",
        "species": "Betelgeusian"
    }
}

write_file_name = "data_file.json"

with open(write_file_name, "w") as write_file:
	# json.dump takes two positional arguments: the json data to write out, and the file-like object to which the bytes will be written 
	json.dump(data, write_file)

print(os.listdir(os.getcwd()))

# you can also write it out as a python native string object
# dumps is pronounced dump-s for dump string
# we aren't writing this to disk however, but storing as it as a dictionary or json string object
json_string = json.dumps(data)

print(json_string)
print(type(json_string))

# change the whitespace indentation of the json string
json_string = json.dumps(data, indent=4)
print(json_string)

# Below, we will be looking at the loads method from json which deserializes the json into a python object
# we are using the load() and loads() (which stands for load-string) methods from json
blackjack_hand = (8, "Q")
encoded_hand = json.dumps(blackjack_hand)
decoded_hand = json.loads(encoded_hand)

print(decoded_hand)

print(blackjack_hand==decoded_hand)
print(type(blackjack_hand))
print(type(decoded_hand))
print(blackjack_hand == tuple(decoded_hand))

# Simple deserialization example.
# say that you have some json data stored in disk that you want to manipulate in memory. You can do this with the context manager.
# This time though, you will open up the existing data_file.json in read mode.

with open("data_file.json", "r") as read_file:
	data = json.load(read_file)

print(data)

json_string = """
{
    "researcher": {
        "name": "Ford Prefect",
        "species": "Betelgeusian",
        "relatives": [
            {
                "name": "Zaphod Beeblebrox",
                "species": "Betelgeusian"
            }
        ]
    }
}
"""

data = json.loads(json_string)

print("This is the data: \n{}".format(data))