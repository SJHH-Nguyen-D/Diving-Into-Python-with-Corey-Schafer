import unittest
import nose2
import os
from importlib import import_module
from fractions import Fraction

target = import_module(".sum", package="my_sum")
sum = target.sum

class TestSum(unittest.TestCase):

    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)

    def test_tuple_int(self):
        """
        Test that it can sum a tuple of integers
        """
        data = (1, 2, 3)
        result = sum(data)
        self.assertEqual(result, 6)

    def test_list_floats(self):
        """
        Test that it can sum list of floats
        """
        data = [1.5, 1.5, 3.0]
        result = sum(data)
        self.assertEqual(result, 6)

    def test_tuple_floats(self):
        """
        test that it can sum a tuple of floats
        """
        data = (1.5, 1.5, 3.0)
        result = sum(data)
        self.assertEqual(result, 6)

    
    def test_sum_list_negative(self):
        """
        Test to see if it can sum list containing negative values
        """
        pass
        data = [1, -2, -3]
        result = sum(data)
        self.assertEqual(result, -4)

    
    def test_sum_tuple_negative(self):
        """
        Test to see if it can sum tuple containing negative values
        """
        data = (1, -2, -3)
        result = sum(data)
        self.assertEqual(result, -4)

    
    def test_sum_type_int_list(self):
        """
        Tests whether the input provided is an iterable elements of type int
        """
        data_list = [1, 2, 3]
        self.assertIsInstance(data_list[0], int)

    
    def test_sum_type_float_list(self):
        """
        Tests whether the input provided is an iterable elements is of type float
        """
        data_list = [1., 2., 3.]
        self.assertIsInstance(data_list[0], float)

    
    def test_list_fraction(self):
        """
        Test that it can sum a list of fractions
        """
        data = [Fraction(1, 4), Fraction(1, 4), Fraction(2, 5)]
        result = sum(data)
        self.assertEqual(result, 1)

    def test_badtypes(self):
        """
        Tests whether an invalid datatype was passed. 
        This test case will now only pass if sum(data) raises a TypeError. 
        You can replace TypeError with any exception type you choose.
        """
        data = "dogs"
        with self.assertRaises(TypeError):
            result = sum(data)

if __name__== "__main__":
    unittest.main() # unittest.main() is the test-runner in unittest