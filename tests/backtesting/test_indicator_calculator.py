import unittest
import pandas as pd
import numpy as np
from src.backtesting.indicator_calculator import _generate_data_fingerprint

class TestIndicatorCalculatorFingerprint(unittest.TestCase):

    def test_generate_data_fingerprint_with_mixed_types(self):
        """
        Test _generate_data_fingerprint handles mixed data types gracefully
        after the change to errors='coerce' for pd.to_numeric.
        """
        data = {
            'numeric_col': [1, 2, 3, 4, 5],
            'string_numeric_col': ['10', '20', '30', '40', '50'],
            'string_text_col': ['apple', 'banana', 'cherry', 'date', 'elderberry'],
            'mixed_col': ['100', 'value', '300', np.nan, '500.5'],
            'float_col': [1.1, 2.2, np.nan, 4.4, 5.5],
            'all_nan_col': [np.nan, np.nan, np.nan, np.nan, np.nan]
        }
        df = pd.DataFrame(data)
        df['datetime_col'] = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        df.set_index('datetime_col', inplace=True)

        fingerprint_all_cols = None
        fingerprint_relevant_cols = None
        
        try:
            # Test with all columns (default behavior if relevant_cols is None)
            fingerprint_all_cols = _generate_data_fingerprint(df)
            
            # Test with a specific list of relevant columns
            relevant_cols = ['numeric_col', 'mixed_col', 'string_text_col', 'non_existent_col']
            fingerprint_relevant_cols = _generate_data_fingerprint(df, relevant_cols=relevant_cols)
            
        except Exception as e:
            self.fail(f"_generate_data_fingerprint raised an unexpected exception: {e}")

        self.assertIsInstance(fingerprint_all_cols, str, "Fingerprint (all_cols) should be a string.")
        self.assertTrue(len(fingerprint_all_cols) > 0, "Fingerprint (all_cols) should not be empty.")
        
        self.assertIsInstance(fingerprint_relevant_cols, str, "Fingerprint (relevant_cols) should be a string.")
        self.assertTrue(len(fingerprint_relevant_cols) > 0, "Fingerprint (relevant_cols) should not be empty.")

        # Optional: Check that fingerprints are different if relevant_cols differ,
        # but the main goal is to ensure no crash and valid output.
        # self.assertNotEqual(fingerprint_all_cols, fingerprint_relevant_cols, 
        #                     "Fingerprints with different column sets should ideally differ.")

        # Test with an empty DataFrame
        empty_df = pd.DataFrame()
        fingerprint_empty = _generate_data_fingerprint(empty_df)
        self.assertEqual(fingerprint_empty, "empty_df")

        # Test with DataFrame having only an index
        df_index_only = pd.DataFrame(index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        fingerprint_index_only = _generate_data_fingerprint(df_index_only)
        self.assertIsInstance(fingerprint_index_only, str)
        self.assertTrue(len(fingerprint_index_only) > 0)
        self.assertNotEqual(fingerprint_index_only, "empty_df")


if __name__ == '__main__':
    unittest.main()
