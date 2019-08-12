import sqlalchemy
import os
from employee import Employee
import sqlite3

# if you want to create an in-memory database, you can specify the connection as such
# working in memory is nice for when you are testing, and for when you don't want to keep deleting a database file over and over
# it just automatically gives you a fresh slate over and over since it is an instance that lives and dies in memory after the instance
# has been closed
conn = sqlite3.connect(":memory:") 

# connect to the database
# If you want your database to persist as a file, you can specify the name of the databse file you want to create and write out
# note that you will not be able to view this employee.db file from the sublime text editor if you don't activate this option in sublime
# conn = sqlite3.connect("employee.db")

# create a cursor that allows us to execute some SQL commands
# we have to use a connection for our cursor
cursor = conn.cursor()

# create an employee table 
cursor.execute(
"""CREATE TABLE employees (
	first text,
	last text,
	pay integer)
	""")


# functions to do database functions such as insert, get, etc

def insert_emp(emp):
	""" inserts an employee into the employee database table """

	# use a context manager conn, which is our connection to the database, for opening, committing and closing
	with conn:
		cursor.execute(
		"INSERT INTO employees VALUES(:first, :last, :pay)", 
		{'first': emp.first, 'last': emp.last, 'pay': emp.pay}
		)


def get_emp_by_lastname(lastname):
	""" select and print """
	cursor.execute("SELECT * FROM employees WHERE last=:last", {'last': lastname})
	return cursor.fetchall()


def update_pay(emp, pay):
	""" update the pay for an employee """
	with conn:
		cursor.execute(
			"""UPDATE employees SET pay=:pay 
			WHERE first=:first AND last=:last""",
			{'first': emp.first, 'last': emp.last, 'pay':emp.pay})


def remove_emp(emp):
	""" removes an employee from the table """
	with conn:
		cursor.execute("""DELETE from employees 
			WHERE first=:first AND last=:last""", 
			{'first': emp.first, 'last': emp.last})



# adding data to this database
# add one observation to the data
emp_1 = Employee('John', 'Doe', 30000)
emp_2 = Employee('Gerald', 'Brathwaight', 100000)
emp_3 = Employee('Bert', 'Steamer', 56000)
emp_4 = Employee('John', 'Doe', 56000)
emp_5 = Employee('Jane', 'Doe', 5000000)

# inserting employees into table with custom functions
insert_emp(emp_1)
insert_emp(emp_2)
insert_emp(emp_3)
insert_emp(emp_4)
insert_emp(emp_5)

# use the select by last name custom method
emps = get_emp_by_lastname('Brathwaight')
print(emps)

# update the pay of an employee
update_pay(emp_5, 56)

# delete an employee record
remove_emp(emp_4)
emps = get_emp_by_lastname('Doe')
print(emps)

# when you are accepting user input for databases, you don't ever want to use braces
# this makes you vulnerable to SQL injection attacks that mess up the database.
# use question marks instead
# cursor.execute("INSERT INTO employees VALUES(?, ?, ?)", (emp_1.first, emp_1.last, emp_1.pay))
# conn.commit()

# another way we can insert entries into the database is through this method
# this is a a better and the more preferred way of inputting entries into the database
# this way uses a dictionary instead of tuple and what we enter here are the values to these keys
# cursor.execute(
# 	"INSERT INTO employees VALUES(:first, :last, :pay)", 
# 	{'first': emp_2.first, 'last': emp_2.last, 'pay': emp_2.pay}
# 	)
# conn.commit()

# This is one way that uses the equestion mark approach to using a select statement
# cursor.execute("SELECT * FROM employees WHERE last=?", ('Brathwaight',))
# print(cursor.fetchall())

# This is another way to declare a select statement using dictionary kvps. 
# cursor.execute("SELECT * FROM employees WHERE last=:last", {'last': 'Doe'})
# print(cursor.fetchall())


# get thenext row in our result and only return that row
# print(cursor.fetchone())

# fetches list of X many entries
# print(cursor.fetchmany(5))

# fetch remaining rows
# print(cursor.fetchall())

conn.close()

