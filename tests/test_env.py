import unittest

from env import _is_transient_model_error


class ModelRetryTests(unittest.TestCase):
    def test_connection_and_timeout_errors_are_retryable(self):
        self.assertTrue(_is_transient_model_error(ConnectionError("connection error")))
        self.assertTrue(_is_transient_model_error(TimeoutError("timed out")))
        self.assertTrue(_is_transient_model_error(RuntimeError("Connection reset")))

    def test_bad_requests_are_not_retryable(self):
        self.assertFalse(
            _is_transient_model_error(RuntimeError("400 invalid request"))
        )


if __name__ == "__main__":
    unittest.main()
