import unittest
import re
from datetime import datetime, timezone, timedelta
from src.backtesting.optimization.objective_function_evaluator import SimpleErrorHandler, ErrorResult

class TestObjectiveFunctionEvaluatorErrorHandling(unittest.TestCase):

    def test_error_result_timestamp_utc(self):
        """Test that ErrorResult populates timestamp_utc correctly."""
        # This test directly instantiates ErrorResult to check its default factory,
        # as the fix was about ensuring 'datetime' is defined for the lambda.
        
        error_instance = ErrorResult(
            error_type="TestError",
            message="This is a test error message.",
            traceback_str="No traceback needed for this test."
            # context and suggestions will use their defaults
        )

        self.assertIsInstance(error_instance.timestamp_utc, str)
        
        # Check if the timestamp_utc string is a valid ISO 8601 format
        # A simple regex can do a basic check, or try parsing it.
        # Regex for ISO 8601 like YYYY-MM-DDTHH:MM:SS.ffffff+ZZ:ZZ or Z
        iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
        self.assertIsNotNone(re.match(iso_pattern, error_instance.timestamp_utc),
                             f"Timestamp {error_instance.timestamp_utc} does not match basic ISO pattern.")

        try:
            parsed_timestamp = datetime.fromisoformat(error_instance.timestamp_utc)
        except ValueError:
            self.fail(f"timestamp_utc '{error_instance.timestamp_utc}' is not a valid ISO format string.")

        # Check that the timestamp is recent (e.g., within the last 5 seconds) and has UTC timezone
        self.assertIsNotNone(parsed_timestamp.tzinfo, "Timestamp should have timezone info.")
        self.assertEqual(parsed_timestamp.tzinfo, timezone.utc, "Timestamp should be UTC.")
        
        now_utc = datetime.now(timezone.utc)
        # Allow a small delta for execution time
        self.assertTrue(now_utc - parsed_timestamp < timedelta(seconds=5),
                        f"Timestamp {parsed_timestamp} is not recent (current UTC: {now_utc}).")

    def test_simple_error_handler_creates_valid_timestamp(self):
        """Test SimpleErrorHandler's handle_evaluation_error for timestamp."""
        handler = SimpleErrorHandler()
        test_exception = ValueError("A sample error for testing.")
        test_context = {"info": "test_case"}

        error_report = handler.handle_evaluation_error(test_exception, test_context)

        self.assertIsInstance(error_report, ErrorResult)
        self.assertIsInstance(error_report.timestamp_utc, str)

        try:
            parsed_timestamp = datetime.fromisoformat(error_report.timestamp_utc)
        except ValueError:
            self.fail(f"Handler's ErrorResult.timestamp_utc '{error_report.timestamp_utc}' is not valid ISO format.")

        self.assertEqual(parsed_timestamp.tzinfo, timezone.utc, "Handler's timestamp should be UTC.")
        now_utc = datetime.now(timezone.utc)
        self.assertTrue(now_utc - parsed_timestamp < timedelta(seconds=5),
                        f"Handler's timestamp {parsed_timestamp} is not recent (current UTC: {now_utc}).")

if __name__ == '__main__':
    unittest.main()
