# Small example program that goes into a directory and prints what .jpg files that are available in a particular folder

import os, glob

directory_to_search = "/home/dennis/Pictures"
os.chdir(directory_to_search)
pngcollection = []

import timeit

start_time = timeit.default_timer()

for file in glob.glob("*.png"):
	pngcollection.append(file)

stop_time = timeit.default_timer()
total_time = stop_time - start_time

print("I have {} .png files in my Pictures folder".format(len(pngcollection)))
print("It took {0:.2}s to run the procedure".format(total_time))