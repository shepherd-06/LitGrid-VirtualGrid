import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        Applies per month rolling averages and exponential smoothing to the data.
        Args:
            df (DataFrame): The DataFrame to analyze.
        Returns:
            DataFrame: The DataFrame with added trend analysis columns.
        """
        # Resample the DataFrame to monthly frequency taking the sum for each month
        monthly_df = df.resample('ME').sum()

        # Apply a rolling average with a window of 1 month
        monthly_df['monthly_rolling_avg'] = monthly_df['value'].rolling(
            window=1).mean()

        # Apply exponential smoothing on the resampled data
        monthly_df['exp_smooth'] = monthly_df['value'].ewm(
            span=1, adjust=False).mean()

        return monthly_df

    def plot_trends(self, actual_df, planned_df, projected_df, title, ylabel):
        """
        Plots the trends of actual, planned, and projected electricity consumption data on a single graph using seaborn.
        """
        plt.figure(figsize=(14, 7))

        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Plot actual consumption using seaborn
        sns.lineplot(data=actual_df, x=actual_df.index, y='monthly_rolling_avg',
                     label='Actual Consumption', color='blue', linewidth=2.5)

        # Plot planned consumption
        sns.lineplot(data=planned_df, x=planned_df.index, y='monthly_rolling_avg',
                     label='Planned Consumption', color='red', linestyle='--', linewidth=2.5)

        # Plot projected consumption
        sns.lineplot(data=projected_df, x=projected_df.index, y='monthly_rolling_avg',
                     label='Projected Consumption', color='green', linestyle=':', linewidth=2.5)

        # Improve the x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # Rotate date labels for better readability
        plt.xticks(rotation=45)

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()

        # Optional: Despine the plot to remove the top and right borders for a cleaner look
        sns.despine()

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
        self.plot_trends(actual_df, planned_df, projected_df,
                         'Electricity Consumption Trends', 'Electricity Consumption (MW)')


if __name__ == "__main__":
    # Make sure to create and configure this according to your environment
    redis_connector = RedisConnector().get_connection()
    trend_analysis = ConsumptionTrendAnalysis(redis_connector)
    trend_analysis.perform_analysis()
