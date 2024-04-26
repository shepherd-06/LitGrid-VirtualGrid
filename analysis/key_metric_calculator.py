import pandas as pd
from pre_processing.key_mappings import KEY_STRUCTURE
from pre_processing.data_processor import _DataProcessor
from pre_processing.data_transformer import DataTransformer
from pre_processing.redis_con import RedisConnector
from tabulate import tabulate


class KeyMetricsCalculator:
    def __init__(self, redis_connector):
        """
        Initialize the calculator with a Redis connector.
        Args:
            redis_connector: An instance of the RedisConnector.
        """
        self.data_processor = _DataProcessor(redis_connector=redis_connector)

    def fetch_and_process_data(self, key_type):
        """
        Fetches data by key type from Redis and transforms it to a DataFrame.
        Args:
            key_type (str): The key type to fetch data for.
        Returns:
            DataFrame: A pandas DataFrame containing the transformed data.
        """
        raw_data = self.data_processor.fetch_data_by_keytype(key_type)
        transformer = DataTransformer(raw_data)
        return transformer.transform_to_dataframe()

    def calculate_metrics(self, df):
        """
        Calculates key metrics for the given DataFrame.
        Args:
            df (DataFrame): The DataFrame to calculate metrics for.
        Returns:
            dict: A dictionary containing the key metrics.
        """
        metrics = {
            'mean': df['value'].mean(),
            'median': df['value'].median(),
            'std_dev': df['value'].std(),
            'min': df['value'].min(),
            'max': df['value'].max()
        }
        return metrics

    def run(self):
        """
        Runs the metrics calculation for each key in KEY_STRUCTURE and prints the results.
        """
        results = []
        for description, key in KEY_STRUCTURE.items():
            print(f"Calculating metrics for {description}...")
            df = self.fetch_and_process_data(key)
            metrics = self.calculate_metrics(df)
            results.append([description] + list(metrics.values()))

        headers = ["Description", "Mean (MW)", "Median (MW)",
                   "Standard Deviation (MW)", "Minimum (MW)", "Maximum (MW)"]
        print(tabulate(results, headers=headers, tablefmt="pretty"))


# Usage example
if __name__ == "__main__":
    redis_connector = RedisConnector().get_connection()
    metrics_calculator = KeyMetricsCalculator(redis_connector)
    metrics_calculator.run()
