import multiprocessing

def square(n):
    result = n**2
    print("The number %d squares to %d" % (n, result))

square(2)