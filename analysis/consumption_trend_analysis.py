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

    def analyze_biweekly_trends(self, df):
        """
        Applies rolling averages and exponential smoothing over 15-day intervals to the data.
        Args:
            df (DataFrame): The DataFrame to analyze.
        Returns:
            DataFrame: The DataFrame with added trend analysis columns for 15-day intervals.
        """
        # Resample the DataFrame to 15-day frequency taking the sum for each interval
        biweekly_df = df.resample('15D').sum()

        # Apply a rolling average with a window of 1 interval (15 days)
        biweekly_df['biweekly_rolling_avg'] = biweekly_df['value'].rolling(
            window=1).mean()

        # Apply exponential smoothing on the resampled data
        biweekly_df['exp_smooth'] = biweekly_df['value'].ewm(
            span=1, adjust=False).mean()

        return biweekly_df

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
        weekly_df['weekly_rolling_avg'] = weekly_df['value'].rolling(
            window=1).mean()

        return weekly_df

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

    def plot_biweekly_trends(self, actual_df, planned_df, projected_df, title, ylabel):
        """
        Plots the trends of actual, planned, and projected electricity consumption data on a single graph using seaborn.
        """
        plt.figure(figsize=(14, 7))

        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Plot actual consumption using seaborn
        sns.lineplot(data=actual_df, x=actual_df.index, y='biweekly_rolling_avg',
                     label='Actual Consumption', color='blue', linewidth=2.5)

        # Plot planned consumption
        sns.lineplot(data=planned_df, x=planned_df.index, y='biweekly_rolling_avg',
                     label='Planned Consumption', color='red', linestyle='--', linewidth=2.5)

        # Plot projected consumption
        sns.lineplot(data=projected_df, x=projected_df.index, y='biweekly_rolling_avg',
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

    def plot_monthly_trends(self, actual_weekly_df, planned_weekly_df, projected_weekly_df, month_year_str, title, ylabel):
        """
        Plots the weekly rolling averages for a specific month.
        Args:
            weekly_df (DataFrame): The DataFrame with the weekly data to plot.
            month_year_str (str): The month and year in the format 'MM-YYYY' for the title.
            title (str): The title of the plot.
            ylabel (str): The label for the y-axis.
        """
        plt.figure(figsize=(10, 5))

        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Plot weekly rolling average
        # Plot actual consumption using seaborn
        sns.lineplot(data=actual_weekly_df, x=actual_weekly_df.index, y='weekly_rolling_avg',
                     label='Actual Rolling Average', color='blue', linewidth=2.5)

        # Plot planned consumption
        sns.lineplot(data=planned_weekly_df, x=planned_weekly_df.index, y='weekly_rolling_avg',
                     label='Planned Consumption', color='red', linestyle='--', linewidth=2.5)

        # Plot projected consumption
        sns.lineplot(data=projected_weekly_df, x=projected_weekly_df.index, y='weekly_rolling_avg',
                     label='Projected Consumption', color='green', linestyle=':', linewidth=2.5)

        # Rotate date labels for better readability
        plt.xticks(rotation=45)

        plt.title(f'{title} - {month_year_str}')
        plt.xlabel('Week')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()

        # Optional: Despine the plot
        sns.despine()

        plt.show()

    def perform_analysis(self, month_year_str=None, months_range=1, biweekly=False):
        """
        Performs the complete analysis by fetching, transforming, and analyzing data.
        """
        if months_range > 2:
            months_range = 2

        actual_df = self.fetch_and_transform_data(
            "electricity_consumption_actual")
        planned_df = self.fetch_and_transform_data(
            "electricity_consumption_planned")
        projected_df = self.fetch_and_transform_data(
            "electricity_consumption_projected")

        if month_year_str is None:
            # Analyze trends
            if biweekly:
                actual_biweekly_df = self.analyze_biweekly_trends(actual_df)
                planned_biweekly_df = self.analyze_biweekly_trends(planned_df)
                projected_biweekly_df = self.analyze_biweekly_trends(
                    projected_df)

                # Plot actual consumption trends

                self.plot_biweekly_trends(actual_biweekly_df, planned_biweekly_df, projected_biweekly_df,
                                          'BiWeekly Electricity Consumption Trends', 'Electricity Consumption (MW)')
            else:
                actual_df = self.analyze_trends(actual_df)
                planned_df = self.analyze_trends(planned_df)
                projected_df = self.analyze_trends(projected_df)

                # Plot actual consumption trends

                self.plot_trends(actual_df, planned_df, projected_df,
                                 'Electricity Consumption Trends', 'Electricity Consumption (MW)')
        else:
            actual_weekly_df = self.analyze_monthly_trends(
                actual_df, month_year_str, months_range)
            planned_weekly_df = self.analyze_monthly_trends(
                planned_df, month_year_str, months_range)
            projected_weekly_df = self.analyze_monthly_trends(
                projected_df, month_year_str, months_range)

            self.plot_monthly_trends(actual_weekly_df, planned_weekly_df,  projected_weekly_df, month_year_str,
                                     'Weekly Electricity Consumption Trends', 'Electricity Consumption (MW)')


if __name__ == "__main__":
    # Make sure to create and configure this according to your environment
    redis_connector = RedisConnector().get_connection()
    trend_analysis = ConsumptionTrendAnalysis(redis_connector)
    # trend_analysis.perform_analysis("01-2023", 2)
    trend_analysis.perform_analysis(biweekly=True)
