import pandas as pd
import matplotlib.pyplot as plt
from pre_processing.data_processor import _DataProcessor
from pre_processing.data_transformer import DataTransformer
from pre_processing.redis_con import RedisConnector


class ConsumptionTrendAnalysis:
    def __init__(self, redis_connector):
        """
        Initialize the analysis with a connection to Redis.
        Args:
            redis_connector: An instance of a RedisConnector.
        """
        self.data_processor = _DataProcessor(redis_connector=redis_connector)

    def fetch_and_transform_data(self, key_type):
        """
        Fetches and transforms data from Redis.
        Args:
            key_type (str): Type of key to fetch data for.
        Returns:
            DataFrame: A pandas DataFrame of the transformed data.
        """
        data = self.data_processor.fetch_data_by_keytype(key_type)
        transformer = DataTransformer(data)
        return transformer.transform_to_dataframe()

    def analyze_trends(self, df):
        """
        Applies rolling averages and exponential smoothing to the data.
        Args:
            df (DataFrame): The DataFrame to analyze.
        Returns:
            DataFrame: The DataFrame with added trend analysis columns.
        """
        df['rolling_avg'] = df['value'].rolling(window=7, center=True).mean()
        df['exp_smooth'] = df['value'].ewm(span=7, adjust=False).mean()
        return df

    def plot_trends(self, df, title, ylabel):
        """
        Plots the trends of the electricity consumption data.
        Args:
            df (DataFrame): The DataFrame containing the data to plot.
            title (str): The title of the plot.
            ylabel (str): The label for the y-axis.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['value'],
                 label='Actual Consumption', color='blue', alpha=0.3)
        plt.plot(df.index, df['rolling_avg'],
                 label='7-Day Rolling Average', color='red')
        plt.plot(df.index, df['exp_smooth'],
                 label='Exponential Smoothing', color='green')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def perform_analysis(self):
        """
        Performs the complete analysis by fetching, transforming, and analyzing data.
        """
        actual_df = self.fetch_and_transform_data(
            "electricity_consumption_actual")
        planned_df = self.fetch_and_transform_data(
            "electricity_consumption_planned")
        projected_df = self.fetch_and_transform_data(
            "electricity_consumption_projected")

        # Analyze trends
        actual_df = self.analyze_trends(actual_df)
        planned_df = self.analyze_trends(planned_df)
        projected_df = self.analyze_trends(projected_df)

        # Plot actual consumption trends
        self.plot_trends(
            actual_df, 'Electricity Consumption Trends - Actual', 'Electricity Consumption (MW)')

        # Optionally, plot planned and projected consumption trends
        self.plot_trends(
            planned_df, 'Electricity Consumption Trends - Planned', 'Electricity Consumption (MW)')
        self.plot_trends(
            projected_df, 'Electricity Consumption Trends - Projected', 'Electricity Consumption (MW)')


if __name__ == "__main__":
    # Make sure to create and configure this according to your environment
    redis_connector = RedisConnector().get_connection()
    trend_analysis = ConsumptionTrendAnalysis(redis_connector)
    trend_analysis.perform_analysis()
