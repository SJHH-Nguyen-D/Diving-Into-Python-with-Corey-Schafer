import threading 
import time 

def func():
    print("ran")
    time.sleep(1)
    print("done")

x = threading.Thread(target=func)
x.start()

# print amount of active threads
# it should say two because by default there is one thread, and now we are opening up a new thread by declaring the use of a new thread called x and opening it with x.start()
print(f"There are {threading.activeCount()} threads running right now.")

# The print for the above function runs inside the func() function call. The reason this happens is that, when the function func() sleeps or hangs for a count,the scheduler tells another thread to fire while the other one is hanging because we switch threads on a hanging process, thus you see the print statement "There are 2 threads running right now" printed after the "ran" print statement.

"""
Here is an example with a global variable running that will be interacted with when running with different threads
"""

print("#### BREAK ###")

ls = []

def count(n):
    for i in range(1, n+1):
        ls.append(i)
        time.sleep(0.5)

def count2(n):
    for i in range(1, n+1):
        ls.append(i)
        time.sleep(0.02) 

x = threading.Thread(target=count, args=(5,))
x.start()
x.join()
y = threading.Thread(target=count2, args=(5,))
y.start()

y.join()
print("Notice how we don't have a list that prints as intended. There are some serialization issues happening")

"""
We can remedy this to print the full list is that we can use the threading.join() command. WHat the two lines below tells us is that we do not print(ls) until both theads x and y have both stopped running. .join() says that we run all processes on thread x entirely before we move onto the next operations happening on the next thread.
"""

print(ls)