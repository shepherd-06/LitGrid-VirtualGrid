import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pre_processing.data_processor import _DataProcessor
from pre_processing.data_transformer import DataTransformer
from pre_processing.redis_con import RedisConnector
from pre_processing.key_mappings import get_key_by_tags, PLOT_TITLES_BY_CATEGORY


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
        Fetches and transforms data from Redis based on the key type.
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
        Analyzes trends based on a given frequency.
        Args:
            df (DataFrame): The DataFrame to analyze.
            freq (str): Frequency for resampling ('M' for monthly, '15D' for biweekly).
        Returns:
            DataFrame: The DataFrame with added trend analysis columns.
        """
        resampled_df = df.resample(freq).mean()
        resampled_df['rolling_avg'] = resampled_df['value'].rolling(
            window=1).mean()
        resampled_df['exp_smooth'] = resampled_df['value'].ewm(
            span=1, adjust=False).mean()
        return resampled_df

    def analyze_monthly_trends(self, df, month_year_str, months_range):
        """
        Analyzes trends by calculating weekly rolling averages for a specific month and additional months before and after.
        Args:
            df (DataFrame): The DataFrame containing the full dataset.
            month_year_str (str): The month and year in the format 'MM-YYYY' to analyze.
            months_range (int): The number of months to include before and after the specified month.
        Returns:
            DataFrame: The DataFrame with the analyzed data for the specified month range.
        """
        # Parse the month_year_str to datetime object to filter the DataFrame
        target_date = pd.to_datetime(month_year_str, format='%m-%Y')

        # Define the start and end dates by subtracting and adding months_range
        start_date = target_date - pd.DateOffset(months=months_range)
        end_date = target_date + pd.DateOffset(months=months_range)

        # Filter the DataFrame for the specified month range
        monthly_range_df = df[(df.index >= start_date)
                              & (df.index <= end_date)]

        # Resample to weekly frequency and compute rolling averages
        weekly_df = monthly_range_df.resample('W').sum()
        weekly_df['rolling_avg'] = weekly_df['value'].rolling(
            window=1).mean()

        return weekly_df

    def plot_trends(self, dfs, title, column):
        """
        Plots the trends for given data frames.
        Args:
            dfs (dict): Dictionary of data frames with labels as keys and data as values.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(14, 7))
        sns.set(style="whitegrid")

        for label, df in dfs.items():
            sns.lineplot(data=df, x=df.index, y='rolling_avg',
                         label=f'{label} Trends')
            if 'outlier' in df.columns:
                outliers = df[df['outlier']]
                sns.scatterplot(x=outliers.index, y='value', data=outliers, color='red',
                                label=f'{label} Outliers', s=50, marker='o', edgecolor='black')

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(column)
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
            "Choose analysis type ('biweekly', 'monthly', 'specific'): ").strip().lower()

        if analysis_type not in ['biweekly', 'monthly', 'specific']:
            print(
                "Invalid input. Please choose 'biweekly', 'monthly', or 'specific'.")
            return

        tag = input(
            "Enter category (consumption, production, source, storage): ")
        tags = get_key_by_tags(tag)
        data_frames = {}
        title = PLOT_TITLES_BY_CATEGORY.get(tag, "Figure: [no title]")
        month_year_str = None

        for description, key in tags.items():
            print(f"Fetching and processing data for {description}")
            df = self.fetch_and_transform_data(key)
            if analysis_type == 'biweekly':
                analyzed_df = self.analyze_trends(df, '15D')
            elif analysis_type == 'monthly':
                analyzed_df = self.analyze_trends(df, 'ME')
            elif analysis_type == 'specific':
                if month_year_str is None:
                    month_year_str = input(
                        "Enter the month and year (MM-YYYY): ")
                    months_range = int(input(
                        "Enter the range of months to include before and after the specified month (1-3): "))
                analyzed_df = self.analyze_monthly_trends(
                    df, month_year_str, months_range)

            data_frames[description] = analyzed_df

        self.plot_trends(data_frames, title, f"{tag.capitalize()} (MW)")


if __name__ == "__main__":
    redis_connector = RedisConnector().get_connection()
    trend_analysis = ConsumptionTrendAnalysis(redis_connector)
    trend_analysis.perform_analysis()
