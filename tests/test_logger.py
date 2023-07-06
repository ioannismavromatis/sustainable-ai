import logging
import unittest

from utils import get_logger, set_level, setup_custom_logger


class TestLoggerFunctions(unittest.TestCase):
    def test_get_logger(self):
        # Test if a logger is returned
        logger = get_logger("test_logger_get")
        self.assertIsInstance(logger, logging.Logger)

    def test_set_level(self):
        # Test if the logging level is correctly set
        custom_logger = get_logger("test_logger_set")
        custom_logger = set_level("test_logger_set", "debug")
        self.assertEqual(custom_logger.getEffectiveLevel(), logging.DEBUG)

        # Test invalid logging level
        with self.assertRaises(ValueError):
            set_level("test_logger_set", "INVALID_LEVEL")

    def test_setup_custom_logger(self):
        # Test creating a new logger with a unique name
        custom_logger = get_logger("test_custom_logger")
        self.assertIsInstance(custom_logger, logging.Logger)

        # Test creating a logger that already exists raises an error
        with self.assertRaises(RuntimeError):
            setup_custom_logger("test_custom_logger", "INFO")

    def test_logger_format(self):
        # Test if the logger format is set correctly
        logger = get_logger("test_logger_format")
        handler = logger.handlers[0]
        formatter = handler.formatter
        self.assertEqual(formatter.datefmt, "%d-%b-%y %H:%M:%S")


if __name__ == "__main__":
    unittest.main()
