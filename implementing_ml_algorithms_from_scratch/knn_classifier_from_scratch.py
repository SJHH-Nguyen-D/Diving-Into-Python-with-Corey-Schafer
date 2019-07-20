import math
import numpy as np 



######################################################################
############### K-Nearest-Neighbours Classifier ######################
######################################################################
######################################################################


def euclidean_distance(X_train, X_test, y_train, y_test):
    """ The euclidean distance implemenation #1 """
    distance = 0
    for x1, x2, y1, y2 in zip(X_train, X_test, y_train, y_test):
        distance = distance + (x1 - x2)**2 - (y1 - y2)**2
    distance = math.sqrt(distance)
    return distance


def e_distance(X1, X2):
    """ The euclidean distance implemenation #2 """
    distance = 0
    for x1, x2 in zip(X1, X2):
        distance = distance + (x1 - x2)**2
    distance = math.sqrt(distance)
    return distance


def eu_dist(X1, X2):
    """ The euclidean distance implemenation #3 """
    return math.sqrt(sum(math.pow(x1-x2, 2) for x1,x2 in zip(X1, X2)))


e_dist = e_distance(dog_heights_train, dog_heights_test)
eu_dist = eu_dist(dog_heights_train, dog_heights_test)
print("This is the point by point implementation of euclidean distance: {}".format(e_dist))
print("This is the other point by point implementation of euclidean distance: {}".format(eu_dist))


def manhattan_distance(X_train, X_test, y_train, y_test):
    """ The Manhattan/City block distance between two collections of numbers """
    distance = 0
    for x1, x2, y1, y2 in zip(X_train, X_test, y_train, y_test):
        distance = distance + abs(x1 - x2) + abs(y1 - y2)
    return distance


print("This is the manhattan distance: {},\nThis is the euclidean_distance: {}\n".format(
    manhattan_distance(dog_heights_train, dog_weights_train, dog_heights_test, dog_weights_test),
    euclidean_distance(dog_heights_train, dog_weights_train, dog_heights_test, dog_weights_test)
    )
)


def k_nearest_neighbours_classification():
    pass