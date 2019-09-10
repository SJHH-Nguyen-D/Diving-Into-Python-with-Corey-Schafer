import pandas as pd 

orders = pd.read_table('http://bit.ly/chiporders')

print(orders.head())

orders.item_name = orders.item_name.str.upper()

# print(orders.item_name)

print(orders[orders.item_name.str.contains('CHICKEN')])
