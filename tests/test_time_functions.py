import unittest

from utils import format_time, plot_time


class TestTimeFunctions(unittest.TestCase):
    def test_format_time(self):
        # Test cases (input_seconds, expected_output)
        test_cases = [
            (0, "0ms"),
            (1, "1s"),
            (61, "1m1s"),
            (3601, "1h1s"),
            (86401, "1D1s"),
            (1000.56, "16m40s"),
            (3600, "1h"),
        ]

        # Assert that format_time returns the expected output for each test case
        for input_seconds, expected_output in test_cases:
            self.assertEqual(format_time(input_seconds), expected_output)

    def test_plot_time(self):
        # Test cases (input_string, expected_output_ms)
        test_cases = [
            ("0ms", 0),
            ("1s", 1000),
            ("1m1s", 61000),
            ("1h1s", 3601000),
            ("1D1s", 86401000),
            ("16m40s", 1000000),
            ("1h", 3600000),
        ]

        # Assert that plot_time returns the expected output in milliseconds for each test case
        for input_string, expected_output_ms in test_cases:
            self.assertEqual(plot_time(input_string), expected_output_ms)

    def test_plot_time_invalid_format(self):
        # Test that plot_time raises ValueError for an invalid time format
        with self.assertRaises(ValueError):
            plot_time("1x")


if __name__ == "__main__":
    unittest.main()
