import pandas as pd
from pre_processing.redis_con import RedisConnector
from pre_processing.data_processor import _DataProcessor
from pre_processing.data_transformer import DataTransformer


class DataHandler:
    def __init__(self, redis_key):
        self.redis_con = RedisConnector().get_connection()
        self.data = self.load_data(redis_key)

    def load_data(self, redis_key):
        """Load data from a JSON file into a DataFrame."""
        data = _DataProcessor(redis_connector=self.redis_con).fetch_data_by_keytype(redis_key)
        transformer = DataTransformer(data)
        df = transformer.transform_to_dataframe()
        
        # Ensure index is datetime and sort by index
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Resample to hourly frequency and fill missing values
        df = df.resample('h').asfreq()  # Use 'h' instead of 'H'
        df = df.ffill()  # Use ffill() method directly
        
        return df

    def segment_data(self, train_size=0.6):
        """Segment data into training and testing datasets."""
        self.data.sort_index(inplace=True)
        cutoff = int(len(self.data) * train_size)
        training_data = self.data.iloc[:cutoff]
        testing_data = self.data.iloc[cutoff:]
        return training_data, testing_data


def main():
    """
    for test.
    """
    redis_key = 'electricity_consumption_actual'
    data_handler = DataHandler(redis_key=redis_key)
    train_data, test_data = data_handler.segment_data()

    # Continue with your analysis or modeling
    print("Data Loaded and Segmented.")
    print(len(train_data), len(test_data))


if __name__ == "__main__":
    main()
