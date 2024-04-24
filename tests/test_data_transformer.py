import unittest
import pandas as pd
from datetime import datetime
from pre_processing.data_transformer import DataTransformer


class TestDataTransformer(unittest.TestCase):
    def test_transform_to_dataframe(self):
        # Mock data as it might come from Redis
        mock_data = {
            b'2023-01-01 00:00:00': b'1000',
            b'2023-01-01 01:00:00': b'1010',
            b'2023-01-01 02:00:00': b'990'
        }

        # Initialize the DataTransformer with the mocked data
        transformer = DataTransformer(mock_data)

        # Perform the transformation
        df = transformer.transform_to_dataframe()

        # Check if DataFrame is correctly formatted
        self.assertIsInstance(
            df, pd.DataFrame, "Output should be a pandas DataFrame.")
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex),
                        "Index should be of type DatetimeIndex.")
        self.assertEqual(df.at[datetime(2023, 1, 1, 0, 0), 'value'],
                         1000.0, "Value conversion or assignment error.")
        self.assertEqual(df.at[datetime(2023, 1, 1, 1, 0), 'value'],
                         1010.0, "Value conversion or assignment error.")
        self.assertEqual(df.at[datetime(2023, 1, 1, 2, 0), 'value'],
                         990.0, "Value conversion or assignment error.")


if __name__ == '__main__':
    unittest.main()
