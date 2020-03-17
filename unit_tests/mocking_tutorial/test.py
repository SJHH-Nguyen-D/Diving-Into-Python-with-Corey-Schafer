import unittest
from unittest.mock import Mock, patch
from requests.exceptions import Timeout
import my_calendar
import datetime

class TestCalendar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nsetUpClass\n")

    def setUp(self):
        print("\nsetUp")

    @patch("__main__.my_calendar.requests", autospec=True)
    def test_get_holidays_retry(self, mock_requests):
        # test first whether or not we have a timeout
        # here we use patch in a context manager as opposed to a decorator
        mock_requests.get.side_effect = Timeout
        with self.assertRaises(Timeout) as error:
            mock_requests.get(error)
            mock_requests.get.assert_called_once()

    def test_is_weekday(self):
        # today
        # tuesday = datetime.datetime(year=2019, month=1, day=1)
        with patch("__main__.my_calendar", autospec=True) as calendar: # mock object's name is calendar
            self.assertEqual(calendar.is_weekday(), True, "Today is not a weekday")
            calendar.assert_called_once()

    def tearDown(self):
        print("\ntearDown")

    @classmethod
    def tearDownClass(cls):
        print("\ntearDownClass")


if __name__ == "__main__":
    unittest.main()
