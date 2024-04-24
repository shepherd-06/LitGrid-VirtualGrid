import pandas as pd
from datetime import datetime
from pre_processing.data_processor import _DataProcessor
import matplotlib.pyplot as plt


class DataTransformer:
    def __init__(self, data):
        """
        Initializes the DataTransformer with Redis data.

        Args:
            data (dict): A dictionary with keys as timestamps and values as measurements.
        """
        self.data = data

    def transform_to_dataframe(self):
        """
        Transforms the raw data into a pandas DataFrame.

        Returns:
            DataFrame: A pandas DataFrame with time as the index and a column for measurements.
        """
        # Initialize a list to store converted data
        records = []
        for timestamp, value in self.data.items():
            # Ensure that value is a string and decode if necessary
            timestamp = timestamp.decode(
                'utf-8') if isinstance(timestamp, bytes) else timestamp
            value = value.decode(
                'utf-8') if isinstance(value, bytes) else value
            # Parse timestamp directly from the key
            timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S')
            records.append({'timestamp': timestamp, 'value': float(value)})

        # Convert the list of records to DataFrame
        df = pd.DataFrame(records)
        # Set the timestamp as the DataFrame index
        df.set_index('timestamp', inplace=True)
        return df


def test_function():
    """
    Function to check and test the functionality of this data_transformer class.
    """
    data = _DataProcessor().fetch_data_by_keytype(
        "electricity_consumption_projected")
    data_1 = DataTransformer(data)
    df = data_1.transform_to_dataframe()
    total_actual_consumption = df['value'].sum()
    print("Total Actual Electricity Consumption:", total_actual_consumption)

    df['value'].plot(title='Actual Electricity Consumption Over Time')
    plt.xlabel('Date')
    plt.ylabel('Consumption (MW)')
    plt.show()


if __name__ == "__main__":
    test_function()
