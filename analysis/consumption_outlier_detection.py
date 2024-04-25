import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pre_processing.data_processor import _DataProcessor
from pre_processing.redis_con import RedisConnector
from pre_processing.data_transformer import DataTransformer


class OutlierDetector:
    def __init__(self, actual_df, planned_df, projected_df, threshold=3):
        """
        Initializes the OutlierDetector with the specified data for actual, planned, and projected consumption.

        Args:
            actual_df, planned_df, projected_df (pd.DataFrame): DataFrames containing the consumption data.
            threshold (float): The Z-score threshold to identify outliers.
        """
        self.actual_df = actual_df
        self.planned_df = planned_df
        self.projected_df = projected_df
        self.threshold = threshold

    def detect_outliers(self, df, column_name):
        """
        Detects outliers in the specified DataFrame using the Z-score method.

        Args:
            df (pd.DataFrame): The DataFrame to detect outliers in.
            column_name (str): The column name to detect outliers for.

        Returns:
            df (pd.DataFrame): The DataFrame with an 'outlier' column indicating outliers.
        """
        z_scores = np.abs(stats.zscore(df[column_name]))
        df['outlier'] = z_scores > self.threshold
        return df

    def plot_outliers(self):
        """
        Plots the actual, planned, and projected consumption data, highlighting the outliers.
        """
        # Set the aesthetic style of the plots
        sns.set(style="whitegrid")

        # Define a figure
        plt.figure(figsize=(14, 7))

        # Plot actual consumption and outliers
        sns.lineplot(data=self.actual_df, x=self.actual_df.index,
                     y='value', label='Actual Consumption', color='blue', alpha=0.5)
        self.highlight_outliers(self.actual_df, 'Actual Outliers', 'blue')

        # Plot planned consumption and outliers
        sns.lineplot(data=self.planned_df, x=self.planned_df.index, y='value',
                     label='Planned Consumption', color='orange', alpha=0.5)
        self.highlight_outliers(self.planned_df, 'Planned Outliers', 'orange')

        # Plot projected consumption and outliers
        sns.lineplot(data=self.projected_df, x=self.projected_df.index,
                     y='value', label='Projected Consumption', color='green', alpha=0.5)
        self.highlight_outliers(
            self.projected_df, 'Projected Outliers', 'green')

        # Final plot adjustments
        plt.title('Outlier Detection for Electricity Consumption')
        plt.xlabel('Date')
        plt.ylabel('Consumption (MW)')
        plt.legend()
        plt.tight_layout()
        sns.despine()
        plt.show()

    def highlight_outliers(self, df, label, color):
        """
        Highlights outliers on the plot.
        """
        outliers = df[df['outlier']]
        sns.scatterplot(data=outliers, x=outliers.index, y='value',
                        label=label, color=color, s=100, marker='o', alpha=0.7)


# Usage with your actual data
if __name__ == "__main__":
    redis_connector = RedisConnector().get_connection()
    data_processor = _DataProcessor(redis_connector=redis_connector)

    actual_data = data_processor.fetch_data_by_keytype(
        "electricity_consumption_actual")
    transformer = DataTransformer(actual_data)
    actual_df = transformer.transform_to_dataframe()

    planned_data = data_processor.fetch_data_by_keytype(
        "electricity_consumption_planned")
    transformer = DataTransformer(planned_data)
    planned_df = transformer.transform_to_dataframe()

    projected_data = data_processor.fetch_data_by_keytype(
        "electricity_consumption_projected")
    transformer = DataTransformer(projected_data)
    projected_df = transformer.transform_to_dataframe()

    detector = OutlierDetector(actual_df, planned_df, projected_df, threshold=3)
    
    # Detect outliers for actual, planned, and projected data
    actual_outliers = detector.detect_outliers(actual_df, 'value')
    planned_outliers = detector.detect_outliers(planned_df, 'value')
    projected_outliers = detector.detect_outliers(projected_df, 'value')
    
    # Plot the data with highlighted outliers
    detector.plot_outliers()
