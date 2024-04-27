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

    def analyze_trends(self, df, freq='ME'):
        """
        General method to analyze trends based on a given frequency and detect outliers.
        """
        resampled_df = df.resample(freq).mean()
        resampled_df['rolling_avg'] = resampled_df['value'].rolling(
            window=1).mean()
        resampled_df['exp_smooth'] = resampled_df['value'].ewm(
            span=1, adjust=False).mean()

        # Outlier detection based on rolling average
        resampled_df['outlier'] = (resampled_df['value'] > resampled_df['rolling_avg'] + 3 * resampled_df['rolling_avg'].std()) | \
            (resampled_df['value'] < resampled_df['rolling_avg'] -
             3 * resampled_df['rolling_avg'].std())
        return resampled_df

    def plot_trends(self, dfs, title):
        """
        Plots the trends for given data frames, safely handling the outlier plotting.
        """
        plt.figure(figsize=(14, 7))
        sns.set(style="whitegrid")

        for label, df in dfs.items():
            sns.lineplot(data=df, x=df.index, y='rolling_avg',
                         label=f'{label} Rolling Avg')
            if 'outlier' in df.columns:
                outliers = df[df['outlier']]
                sns.scatterplot(x=outliers.index, y='value', data=outliers, color='red',
                                label=f'{label} Outliers', s=50, marker='o', edgecolor='black')

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Electricity Consumption (MW)')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        sns.despine()
        plt.show()

    def perform_analysis(self):
        """
        Performs the complete analysis by fetching, transforming, and analyzing data based on user input.
        """
        analysis_type = input(
            "Choose analysis type ('biweekly', 'monthly', 'specific month'): ").strip().lower()

        if analysis_type not in ['biweekly', 'monthly', 'specific month']:
            print(
                "Invalid input. Please choose 'biweekly', 'monthly', or 'specific month'.")
            return

        key_type = "electricity_consumption_actual"  # Adjust as necessary
        df = self.fetch_and_transform_data(key_type)

        if analysis_type == 'biweekly':
            analyzed_df = self.analyze_trends(df, '15D')
            self.plot_trends({'Biweekly Analysis': analyzed_df},
                             'Biweekly Consumption Trends')

        elif analysis_type == 'monthly':
            analyzed_df = self.analyze_trends(df, 'ME')
            self.plot_trends({'Monthly Analysis': analyzed_df},
                             'Monthly Consumption Trends')

        elif analysis_type == 'specific month':
            # TODO:
            month_year_str = input("Enter the month and year (MM-YYYY): ")
            months_range = int(input(
                "Enter the range of months to include before and after the specified month (1-3): "))
            analyzed_df = self.analyze_monthly_trends(
                df, month_year_str, months_range)
            self.plot_trends({'Specific Month Analysis': analyzed_df},
                             f'Monthly Trends Around {month_year_str}')


if __name__ == "__main__":
    redis_connector = RedisConnector().get_connection()
    trend_analysis = ConsumptionTrendAnalysis(redis_connector)
    trend_analysis.perform_analysis()
