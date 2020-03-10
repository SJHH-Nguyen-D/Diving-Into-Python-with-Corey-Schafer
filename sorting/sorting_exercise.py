# Sorting a list using a lambda function, and indexing into a tuple for the value

planets = [
("Mercury", 2440, 5.43, 0.395),
("Venus", 6052, 5.24, 0.723),
("Earth", 6378, 5.52, 1.000),
("Mars", 3396, 3.93, 1.530),
("Jupiter", 71492, 1.33, 5.210),
("Saturn", 60268, 0.59, 9.551), 
("Uranus", 25559, 1.27, 19.213), 
("Neptune", 24764, 1.64, 30.070)
]

size = lambda planets: planets[1]
density = lambda planets: planets[2]
aus_from_sun = lambda planets: planets[3]

# Sort changes the list itself in place, thus changing the list itself
# Tuples are immutable and therefore cannot be changed or sorted
planets.sort(key=aus_from_sun, reverse=True)

for i in planets:
	print("{}\n".format(i))

# What if you wanted a sorted copy of the list
# You can sort a tuple with a method instead called tuple.sorted()
# You can sort both a tuple AND make a copy of it or an iterable with the .sorted() method

pokemon = ["Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard", "Squirtle", "Wartortle", "Blastoise"]

# print(help(dict))

pokesort = sorted(pokemon)
print(pokesort)

pokedex_entry_num = list(range(1, len(pokemon)+1))

pokemondict = dict(list(zip(pokedex_entry_num, pokemon)))
print(pokemondict)
