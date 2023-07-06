import unittest
from unittest.mock import Mock

from utils import file_name_generator


class TestFileNameGenerator(unittest.TestCase):
    def test_valid_modes_with_get_stats(self):
        # Mock the args object with get_stats as True and tool as an empty string
        args = Mock(get_stats=True, tool="")

        # Check for mode "train"
        self.assertEqual(file_name_generator(args, "train"), "stats_train")

        # Check for mode "test"
        self.assertEqual(file_name_generator(args, "test"), "stats_test")

    def test_valid_modes_without_get_stats(self):
        # Mock the args object with get_stats as False and tool as an empty string
        args = Mock(get_stats=False, tool="")

        # Check for mode "train"
        self.assertEqual(file_name_generator(args, "train"), "nostats_train")

        # Check for mode "test"
        self.assertEqual(file_name_generator(args, "test"), "nostats_test")

    def test_tool_prefix(self):
        # Mock the args object with a non-empty tool string
        args = Mock(get_stats=True, tool="mytool")

        # Check that the tool name is prefixed to the mode
        self.assertEqual(file_name_generator(args, "train"), "mytool_stats_train")

    def test_invalid_mode(self):
        # Mock the args object with get_stats as True and tool as an empty string
        args = Mock(get_stats=True, tool="")

        # Check that ValueError is raised for an invalid mode
        with self.assertRaises(ValueError):
            file_name_generator(args, "invalid_mode")


if __name__ == "__main__":
    unittest.main()
