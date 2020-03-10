import numpy as np 

#################### SELECTION SORT ######################################
def quick_selection(A):
	"""
	Identifies the lowest number in the list, going through all the numbers, and then swaps it into 
	the first position. We start with the first number in the list and set that as the min value and compare it against
	the subsequent numbers in the list, updating the min value as we traverse right-wise down the list.
	Not a fast sorting algorithm because it uses nested loops and only really useful for small datasets (less than 1000 records).
	Run time: O(n**2)
	"""
	# loop through the list to the second to last item of the list
	from timeit import default_timer
	start = default_timer()
	for i in range(0, len(A)-1):
		minIndex=i
		# now iterated through the unsorted part of the list
		for j in range(i+1, len(A)):
			if A[j] < A[minIndex]:
				minIndex = j
		if minIndex != i:
			A[i], A[minIndex] = A[minIndex], A[i]
	end = default_timer()
	print(f"Selection sort took {end-start:.7f}s to run")
	return A


#################### QUICK SORT ######################################
def quick_sort(A):
	""" Good for big lists. 
		Worst case is O(n**2)
		Best case is O(n log(n))
		Performance is largely based on the selected pivot
	"""
	from timeit import default_timer
	start = default_timer()
	A = quick_sort2(A, 0, len(A)-1)
	end = default_timer()
	print(f"Quick sort took {end-start:.7f}s to run")
	return A

def quick_sort2(A, low, high, threshold=1):
	""" recursive function so it calls itself"""
	# if there is less than a certain number of items in the list, use the selection sort method instead
	if high-low < threshold and low < high: # if there is more than one item to be sorted...
		quick_selection(A, low, high)
	elif low < high:
		pivot = partition(A, low, high)
		quick_sort2(A, low, pivot-1) # for all the items left of the pivot, call quicksort2
		quick_sort2(A, pivot+1, high) # for all the items right of the pivot, call quicksort2
	return A

def partition(A, low, high):
	pivotIndex = get_pivot(A, low, high)
	pivotValue = A[pivotIndex]
	A[pivotIndex], A[low] = A[low], A[pivotIndex]
	boarder = low

	for i in range(low, high+1):
		if A[i] < pivotValue:
			boarder +=1
			A[i], A[boarder] = A[boarder], A[i]
	A[low], A[boarder] = A[boarder], A[low]
	return boarder

def get_pivot(A, low, high):
	mid_point = (low + high)//2
	pivot = high
	if A[low] < A[mid_point]:
		if A[mid_point] < A[high]:
			pivot = mid_point
	elif A[low] < A[high]:
		pivot = low
	return pivot


#################### INSERTION SORT ######################################
def insertion_sort(A):
	"""
	Only good for relatively small datasets because it uses a nested loop.
	It runs in O(n**2). Because this type of list moves smaller numbers up to the first, it requires us to 
	iterate through the list and move items backwards through the indices.
	"""
	for i in range(1, len(A)):
		for j in range(i-1, 0, -1): # a step of negative 1 means that we want to move leftward in the list
			if A[j] > A[j+1]:
				A[j], A[j+1] = A[j+1], A[j]
				j -= 1
			else:
				break
	return A

def faster_overwriting_insertion_sort(A):
	""" instead of performing a copying technique like in the insertion swap method above, we simply
	have a place holder variable that is used to compare against each value moving backwards through the list and we
	write it down if the number it compares to is smaller than it."""
	for i in range(1, len(A)):
		curNum = A[i]
		for j in range(i-1, 0, -1):
			if A[j] > curNum:
				A[j+1] = A[j]
			else:
				A[j+1] = curNum
				break

#################### BUBBLE SORT ######################################

def bubble_sort(arr):
	""" relatively slow sorting algorithm """
	from timeit import default_timer
	start_time = default_timer()
	for i in range(0, len(arr)-1):
		for j in range(0, len(arr)-1-i):
			if arr[j] > arr[j+1]:
				arr[j], arr[j+1] = arr[j+1], arr[j]
	end_time = default_timer()
	print(f"Bubble sort took {end_time-start_time:.10f}s to complete")
	return arr


#################### MERGE SORT ######################################
def merge_sort(A):
	""" relatively fast sorting algorithm for large datasets, not necessarily for smaller datasets.
	has O(n*log(n)) runtime complexity because:
		it does log n merge steps because each merge step doubles the list size.
		it does n work for each merge step because it must look at every item.
		So, combining the two: it runs in O(n log(n))
	"""
	from timeit import default_timer
	start = default_timer()
	A = merge_sort2(A, 0, len(A)-1)
	end = default_timer()
	print(f"Merge sort took {end-start:.7f}s to run")
	return A

def merge_sort2(A, first, last):
	# if first < last means that if there is more than 1 element in the list
	if first < last:
		middle = (first + last)//2
		merge_sort2(A, first, middle)
		merge_sort2(A, middle+1, last)
		merge(A, first, middle, last)
	return A

def merge(A, first, middle, last):
	L  = A[first:middle]
	R = A[middle:last+1]
	L.append(99999999) # added to the end of the list to indicate that we have reached the list terminal
	R.append(99999999)
	i = j = 0
	for k in range(first, last+1):
		if L[i] <= R[j]:
			A[k] = L[i]
			i += 1
		else:
			A[k] = R[j]
			j += 1


myList = [10, 50, 2, 8, 13, 1000]
n = len(myList)

print(f"Bubble sorted list : {bubble_sort(myList)}\n")

print(f"Merge sorted list : {merge_sort(myList)}\n")

print(f"Insertion sorted list : {insertion_sort(myList)}\n")

print(f"Selection sorted list : {quick_selection(myList)}\n")

print(f"Quick sorted list : {quick_sort(myList)}\n")
