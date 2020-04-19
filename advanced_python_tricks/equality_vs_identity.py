a = [1, 2, 3]

# b is now POINTING to A and they are the same identical object
b = a

print(f"Is a equivalent to b: {a == b}")
print(f"Is a identical to b: {a is b}")

# c is a DIFFERENT object than A but they have equivalent values
c = [1, 2, 3]
print(f"Is a equivalent to c: {a == c}")
print(f"Is a identical to c: {a is c}")