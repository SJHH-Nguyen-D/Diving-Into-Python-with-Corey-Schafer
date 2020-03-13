from calendar import is_weekday
import unittest
from datetime import datetime
from unittest.mock import Mock


class TestCalendar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nsetUpClass\n")

    def setUp(self):
        self.datetime = Mock()
        self.result = is_weekday()
        self.tuesday = datetime(year=2019, month=1, day=1)
        self.saturday = datetime(year=2019, month=1, day=5)
        print("\nsetUp")

    def test_is_weekday(self):
        # today
        self.assertEqual(self.result, True, "Today is not a weekday")

    def test_is_today_tuesday(self):
        self.datetime.today.return_value = self.tuesday  # mock object
        (self.assertEqual(self.result, datetime.today(),
                          "Today is not a Tuesday"))

    def test_is_today_saturday(self):
        self.datetime.today.return_value = self.tuesday
        (self.assertEqual(self.result, datetime.today(),
                          "Today is not a Saturday"))

    def tearDown(self):
        print("\ntearDown")

    @classmethod
    def tearDownClass(cls):
        print("\ntearDownClass")


if __name__ == "__main__":
    unittest.main()
